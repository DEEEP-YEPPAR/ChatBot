## Work of image and text for query



import streamlit as st
import fitz  # PyMuPDF
import io
import re
import numpy as np
import os
import tempfile
from PIL import Image
from groq import Groq
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import CLIPProcessor, CLIPModel
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Load environment variables
load_dotenv()

class Config:
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    TEXT_EMBEDDER_MODEL = "all-MiniLM-L6-v2"
    CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
    CHUNK_SIZE = 1500
    CHUNK_OVERLAP = 300
    GROQ_MODEL = "llama3-70b-8192"
    TOP_K = 2
    MAX_TOKENS = 300

@st.cache_resource
def load_models():
    return {
        "text_embedder": SentenceTransformer(Config.TEXT_EMBEDDER_MODEL),
        "clip_model": CLIPModel.from_pretrained(Config.CLIP_MODEL_NAME),
        "clip_processor": CLIPProcessor.from_pretrained(Config.CLIP_MODEL_NAME),
        "text_splitter": RecursiveCharacterTextSplitter(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP,
            length_function=len,
            is_separator_regex=False
        ),
        "groq_client": Groq(api_key=Config.GROQ_API_KEY) if Config.GROQ_API_KEY else None
    }

class PDFTypeDetector:
    @staticmethod
    def analyze(content):
        has_text = len(content["texts"]) > 0
        has_images = len(content["images"]) > 0
        
        if has_text and has_images:
            return "multimodal"
        return "text-only" if has_text else "image-only"

class TextProcessor:
    def __init__(self, models):
        self.models = models
        self.splitter = models["text_splitter"]
    
    def normalize_text(self, texts):
        cleaned = []
        for text in texts:
            content = re.sub(r'\s+', ' ', text['content']).strip()
            content = re.sub(r'(?<!\n)\n(?!\n)', ' ', content)
            cleaned.append({**text, 'content': content})
        return cleaned
    
    def create_embeddings(self, texts):
        embeddings = self.models["text_embedder"].encode(
            [t["content"] for t in texts], show_progress_bar=False
        )
        return embeddings.reshape(len(texts), -1) if len(embeddings.shape) == 1 else embeddings

class VisualProcessor:
    def __init__(self, models):
        self.models = models
    
    def process_images(self, images):
        processed = []
        for img in images:
            try:
                inputs = self.models["clip_processor"](
                    images=img["image"], 
                    return_tensors="pt"
                )
                features = self.models["clip_model"].get_image_features(**inputs)
                embedding = features.detach().numpy().squeeze()
                processed.append({
                    **img,
                    "embedding": embedding.reshape(1, -1),
                    "metadata": {
                        "dimensions": f"{img['image'].width}x{img['image'].height}",
                        "page": img["page"]
                    }
                })
            except Exception as e:
                st.error(f"Image processing error: {str(e)}")
        return processed

class KnowledgeGraph:
    def __init__(self):
        self.nodes = []
    
    def add_node(self, node_type, content, embedding, metadata):
        self.nodes.append({
            "type": node_type,
            "content": content,
            "embedding": embedding,
            "metadata": metadata
        })
    
    def semantic_search(self, query_embedding, node_types=None, top_k=3):
        results = []
        query_embed = query_embedding.reshape(1, -1) if len(query_embedding.shape) == 1 else query_embedding
        
        for node in self.nodes:
            if node_types and node["type"] not in node_types:
                continue
            
            node_embed = node["embedding"]
            if len(node_embed.shape) == 1:
                node_embed = node_embed.reshape(1, -1)
            
            similarity = cosine_similarity(query_embed, node_embed)[0][0]
            results.append((similarity, node))
        
        return sorted(results, reverse=True)[:top_k]

class ResponseEngine:
    def __init__(self, groq_client):
        self.client = groq_client
    
    def generate_response(self, prompt, context, response_type):
        system_prompt = self._create_system_message(response_type)
        
        response = self.client.chat.completions.create(
            messages=[{
                "role": "system",
                "content": system_prompt
            }, {
                "role": "user",
                "content": f"Document Context:\n{context}\n\nQuestion: {prompt}"
            }],
            model=Config.GROQ_MODEL,
            temperature=0.3,
            max_tokens=Config.MAX_TOKENS
        )
        return self._format_response(response.choices[0].message.content)
    
    def _create_system_message(self, response_type):
        base = "You are an expert document analyst. Provide:"
        if response_type == "text":
            return f"{base}\n1. Accurate information from the document.\n2. well structured responses with clear sections and required formate\n3. do nor mention the page number in the response\n4. if document does not have the info, then say 'this document does not contain the info related query"
        elif response_type == "image":
            return f"{base}\n1. Image relevance explanation\n2. Connections to document content"
        return f"{base}\n1. Integrated text and visual analysis\n2. Comprehensive explanations"
    
    def _format_response(self, text):
        text = re.sub(r'\n- ', '\n• ', text)
        return re.sub(r'\[ ?Page (\d+) ?\]', r'[Page \1]', text).strip()

class DocumentAnalyzer:
    def __init__(self, models):
        self.models = models
        self.knowledge_graph = KnowledgeGraph()
    
    def process_document(self, file_bytes):
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        content = self._extract_content(doc)
        
        pdf_type = PDFTypeDetector.analyze(content)
        
        if pdf_type in ["text-only", "multimodal"]:
            text_processor = TextProcessor(self.models)
            cleaned_texts = text_processor.normalize_text(content["texts"])
            text_embeddings = text_processor.create_embeddings(cleaned_texts)
            for text, emb in zip(cleaned_texts, text_embeddings):
                self.knowledge_graph.add_node(
                    "text", text["content"], emb,
                    {"page": text["page"], "type": "text"}
                )
        
        if pdf_type in ["image-only", "multimodal"]:
            visual_processor = VisualProcessor(self.models)
            processed_images = visual_processor.process_images(content["images"])
            for img in processed_images:
                self.knowledge_graph.add_node(
                    "image", None, img["embedding"],
                    {"page": img["page"], "image": img["image"]}
                )
        
        return {
            "type": pdf_type,
            "knowledge_graph": self.knowledge_graph,
            "pages": len(doc)
        }
    
    def _extract_content(self, doc):
        texts = []
        images = []
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            texts.extend(self._extract_page_text(page, page_num))
            images.extend(self._extract_page_images(page, page_num))
        
        return {"texts": texts, "images": images}
    
    def _extract_page_text(self, page, page_num):
        raw_text = page.get_text("text")
        chunks = self.models["text_splitter"].split_text(raw_text)
        return [{
            "content": chunk,
            "page": page_num + 1,
            "type": "text"
        } for chunk in chunks]
    
    def _extract_page_images(self, page, page_num):
        img_list = page.get_images(full=True)
        extracted = []
        for img_info in img_list:
            try:
                xref = img_info[0]
                base_image = page.parent.extract_image(xref)
                image = Image.open(io.BytesIO(base_image["image"]))
                extracted.append({
                    "image": image,
                    "page": page_num + 1,
                    "type": "image"
                })
            except Exception as e:
                print(f"Image extraction error: {str(e)}")
        return extracted

def query_classifier(query):
    visual_keywords = {"image", "photo","show" "diagram", "visual", "picture"}
    text_keywords = {"explain", "describe", "what", "how", "why"}
    
    query_words = set(query.lower().split())
    has_visual = len(query_words & visual_keywords) > 0
    has_text = len(query_words & text_keywords) > 0
    
    if has_visual and has_text:
        return "mixed"
    return "visual" if has_visual else "text"

def main():
    st.set_page_config(page_title="Multi-Modal PDF Analyst", layout="wide")
    st.title("📚 PDF Chat Bot")
    st.caption("Advanced PDF analysis with semantic understanding")
    
    models = load_models()
    
    if "analysis" not in st.session_state:
        st.session_state.analysis = None
    if "history" not in st.session_state:
        st.session_state.history = []
    
    with st.sidebar:
        st.header("Document Management")
        uploaded_file = st.file_uploader("Upload PDF Document", type=["pdf"])
        if st.button("Reset Session"):
            st.session_state.clear()
            st.rerun()
    
    if uploaded_file and not st.session_state.analysis:
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
        
        with open(tmp_file.name, "rb") as f:
            analyzer = DocumentAnalyzer(models)
            st.session_state.analysis = analyzer.process_document(f.read())
        
        os.remove(tmp_file.name)
        st.success("Document analysis completed!")
    
    for entry in st.session_state.history:
        with st.chat_message(entry["role"]):
            st.markdown(entry["content"])
            if "images" in entry and entry["images"]:
                cols = st.columns(max(1, min(3, len(entry["images"]))))
                for idx, img in enumerate(entry["images"]):
                    cols[idx].image(img["image"], caption=f"Page {img['page']}")
    
    if prompt := st.chat_input("Ask about the document..."):
        if not st.session_state.analysis:
            st.error("Please upload a PDF document first")
            return
        
        st.session_state.history.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        analysis = st.session_state.analysis
        kg = analysis["knowledge_graph"]
        response_engine = ResponseEngine(models["groq_client"])
        
        with st.spinner("Analyzing query..."):
            query_type = query_classifier(prompt)
            context = []
            visual_results = []
            
            if query_type in ["text", "mixed"]:
                text_embed = models["text_embedder"].encode([prompt])[0]
                text_matches = kg.semantic_search(text_embed, ["text"], Config.TOP_K)
                context.extend([f"Page {match[1]['metadata']['page']}: {match[1]['content']}" 
                              for match in text_matches])
            
            if query_type in ["visual", "mixed"]:
                inputs = models["clip_processor"](text=[prompt], return_tensors="pt", padding=True)
                text_features = models["clip_model"].get_text_features(**inputs)
                visual_embed = text_features.detach().numpy().squeeze()
                visual_matches = kg.semantic_search(visual_embed, ["image"], Config.TOP_K)
                visual_results = [match[1]["metadata"] for match in visual_matches]
            
            response = response_engine.generate_response(
                prompt, "\n".join(context), query_type
            )
        
        with st.chat_message("assistant"):
            st.markdown(response)
            if visual_results:
                cols = st.columns(max(1, min(3, len(visual_results))))
                for idx, result in enumerate(visual_results):
                    cols[idx].image(result["image"], caption=f"Page {result['page']}")
            
            st.session_state.history.append({
                "role": "assistant",
                "content": response,
                "images": visual_results
            })

if __name__ == "__main__":
    main()
    
    
    
    
