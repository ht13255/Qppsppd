import streamlit as st
import arxiv
import fitz  # PyMuPDF
import spacy
import json
import os
import re
import time
from datetime import datetime, timedelta

# --- 상수 및 설정 ---
STORAGE_PATH = "./papers"
OUTPUT_FILENAME = "knowledge_graph_live.json"
OUTPUT_PATH = os.path.join(STORAGE_PATH, OUTPUT_FILENAME)

# --- 핵심 로직 클래스 ---
class KnowledgeGraphGenerator:
    """ArXiv 논문을 처리하여 지식 그래프를 생성하는 백엔드 로직."""
    
    @st.cache_resource
    def load_spacy_model(_self):
        """Streamlit의 캐시를 사용해 spaCy 모델을 효율적으로 로드합니다."""
        with st.spinner("언어 모델(spaCy)을 로딩 중입니다..."):
            try
                model = spacy.load("en_core_web_sm")
                return model
            except OSError:
                st.error("spaCy 모델을 찾을 수 없습니다. 터미널에서 'python -m spacy download en_core_web_sm'를 실행해주세요.")
                st.stop()

    def __init__(self):
        self.nlp = self.load_spacy_model()

    def search_papers(self, keyword, max_results):
        """ArXiv에서 논문 메타데이터를 검색합니다."""
        try:
            search = arxiv.Search(
                query=keyword,
                max_results=max_results,
                sort_by=arxiv.SortCriterion.Relevance
            )
            return list(search.results())
        except Exception as e:
            st.error(f"ArXiv API 검색 중 오류가 발생했습니다: {e}")
            return []

    def process_single_paper(self, paper_result):
        """단일 논문을 다운로드, 분석하고 트리플을 반환합니다."""
        try:
            safe_title = re.sub(r'[\\/*?:"<>|]', "", paper_result.title)
            pdf_path = os.path.join(STORAGE_PATH, f"{safe_title}.pdf")
            if not os.path.exists(pdf_path):
                paper_result.download_pdf(dirpath=STORAGE_PATH, filename=f"{safe_title}.pdf")
            
            paper_info = {"title": paper_result.title, "url": paper_result.entry_id}
            text = self.extract_text_from_pdf(pdf_path)
            if not text:
                return []

            return self.create_triples_from_text(text, paper_info)
        except Exception:
            return [] # 오류 발생 시 빈 리스트 반환

    def extract_text_from_pdf(self, pdf_path):
        try:
            with fitz.open(pdf_path) as doc:
                text = " ".join(page.get_text() for page in doc)
            return re.sub(r'\s+', ' ', text).replace('-\n', '')
        except Exception:
            return ""

    def create_triples_from_text(self, text, paper_info, chunk_size=50000):
        triples = []
        # NLP 모델의 메모리 한계를 고려하여 텍스트를 chunk 단위로 나누어 처리
        for i in range(0, len(text), chunk_size):
            chunk = text[i:i + chunk_size]
            doc = self.nlp(chunk)
            for sent in doc.sents:
                subjects = [tok for tok in sent if "subj" in tok.dep_]
                objects = [tok for tok in sent if "obj" in tok.dep_]
                if subjects and objects:
                    verb = sent.root
                    for sub in subjects:
                        for obj in objects:
                            subject_phrase = " ".join(t.text for t in sub.subtree)
                            object_phrase = " ".join(t.text for t in obj.subtree)
                            triples.append({
                                "subject": subject_phrase,
                                "predicate": verb.lemma_,
                                "object": object_phrase,
                                "source_title": paper_info['title'],
                                "source_url": paper_info['url']
                            })
        return triples

    def append_triples_to_json(self, new_triples):
        """기존 JSON 파일에 새로운 트리플 데이터를 추가합니다."""
        if not new_triples:
            return
        
        existing_data = []
        if os.path.exists(OUTPUT_PATH):
            try:
                with open(OUTPUT_PATH, 'r', encoding='utf-8') as f:
                    existing_data = json.load(f)
            except json.JSONDecodeError:
                pass # 파일이 비어 있으면 무시
        
        existing_data.extend(new_triples)
        
        with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
            json.dump(existing_data, f, ensure_ascii=False, indent=4)

# --- Streamlit UI ---
def main():
    st.set_page_config(page_title="ArXiv 대규모 분석기", layout="wide")
    st.title("🚀 ArXiv 대규모 지식 그래프 생성기")
    st.markdown("API가 허용하는 최대 수준까지 논문을 분석하여 지식 그래프를 생성합니다.")

    with st.sidebar:
        st.header("⚙️ 분석 설정")
        keyword = st.text_input("검색 키워드", "transformer model")
        max_papers = st.number_input("최대 논문 수", min_value=1, max_value=30000, value=100)
        
        st.warning(
            "⚠️ **주의:** 100개 이상의 논문을 처리하는 것은 매우 긴 시간(수십 분 ~ 수 시간)이 소요될 수 있습니다. "
            "로컬 환경에서 실행하는 것을 권장합니다."
        )
        
        start_button = st.button("대규모 분석 시작", type="primary")

    if start_button:
        # 폴더 및 결과 파일 초기화
        os.makedirs(STORAGE_PATH, exist_ok=True)
        if os.path.exists(OUTPUT_PATH):
            os.remove(OUTPUT_PATH)

        generator = KnowledgeGraphGenerator()
        
        with st.status("ArXiv에서 논문 목록을 가져오는 중...", expanded=True) as status:
            papers = generator.search_papers(keyword, max_papers)
            if not papers:
                status.update(label="검색 결과 없음", state="error")
                st.error("해당 키워드의 논문을 찾을 수 없습니다.")
                st.stop()
            
            status.update(label=f"총 {len(papers)}개의 논문 발견! 분석을 시작합니다.", state="running")

        st.info(f"'{keyword}'에 대해 총 {len(papers)}개의 논문 분석을 시작합니다. 모든 결과는 실시간으로 저장됩니다.")
        
        # 진행 상황 표시를 위한 UI 요소
        progress_bar = st.progress(0)
        progress_text = st.empty()
        
        start_time = time.time()
        total_triples_found = 0

        for i, paper in enumerate(papers):
            # ETA 계산
            elapsed_time = time.time() - start_time
            papers_processed = i + 1
            avg_time_per_paper = elapsed_time / papers_processed
            remaining_papers = len(papers) - papers_processed
            eta_seconds = remaining_papers * avg_time_per_paper
            eta_str = str(timedelta(seconds=int(eta_seconds)))

            # 진행 상황 업데이트
            progress_bar.progress(papers_processed / len(papers))
            progress_text.markdown(
                f"**진행률: {papers_processed}/{len(papers)}** | "
                f"**찾은 트리플: {total_triples_found}개** | "
                f"**예상 남은 시간: {eta_str}**"
                f"<br>📄 현재 분석 중: *{paper.title[:80]}...*",
                unsafe_allow_html=True
            )

            # 핵심 로직 실행
            triples = generator.process_single_paper(paper)
            if triples:
                total_triples_found += len(triples)
                generator.append_triples_to_json(triples)
        
        progress_bar.progress(1.0)
        progress_text.success(f"✅ 분석 완료! 총 {total_triples_found}개의 트리플을 찾아 '{OUTPUT_PATH}'에 저장했습니다.")
        
        # 세션 상태에 최종 결과 로드
        if os.path.exists(OUTPUT_PATH):
            with open(OUTPUT_PATH, 'r', encoding='utf-8') as f:
                st.session_state['results'] = json.load(f)

    # 분석이 완료되거나 기존 결과가 있을 때 표시
    if 'results' in st.session_state and st.session_state['results']:
        st.header("📊 최종 분석 결과")
        
        results_data = st.session_state['results']
        json_string = json.dumps(results_data, indent=4, ensure_ascii=False)
        
        st.download_button(
            label=f"📄 JSON 파일 다운로드 ({len(results_data)}개 트리플)",
            data=json_string,
            file_name=f"kg_{keyword.replace(' ', '_')}.json",
            mime="application/json",
        )

        st.dataframe(results_data, height=600)

if __name__ == "__main__":
    main()
