"""
ToxiRAG Streamlit Application
Main interface for liver cancer toxicity prediction and experiment planning.
"""

import os
import streamlit as st
import tempfile
import asyncio
from pathlib import Path
from typing import Optional, List, Dict, Any

# Local imports (will be created)
try:
    from ingest.ingest_local import ingest_markdown_file
    from retriever.retriever import retrieve_relevant_docs
    from llm.agentic_pipeline import create_agentic_response
except ImportError:
    st.error("Missing local modules. Please ensure ingest/, retriever/, and llm/ modules are implemented.")

def setup_page_config():
    """Configure Streamlit page settings."""
    st.set_page_config(
        page_title="ToxiRAG - 肝癌毒性预测系统",
        page_icon="🧬",
        layout="wide",
        initial_sidebar_state="expanded"
    )

def setup_sidebar() -> Dict[str, Any]:
    """Setup sidebar with API keys and model selection."""
    st.sidebar.title("🔧 配置设置")
    
    # API Keys section
    st.sidebar.subheader("API 密钥")
    openai_api_key = st.sidebar.text_input(
        "OpenAI API Key",
        type="password",
        value=os.getenv("OPENAI_API_KEY", ""),
        help="用于 GPT-5 Nano 和 OpenAI 嵌入"
    )
    
    google_api_key = st.sidebar.text_input(
        "Google API Key", 
        type="password",
        value=os.getenv("GOOGLE_API_KEY", ""),
        help="用于 Gemini 2.5 Flash"
    )
    
    # Model selection
    st.sidebar.subheader("模型选择")
    llm_provider = st.sidebar.selectbox(
        "LLM 提供商",
        ["openai", "gemini"],
        help="选择主要的语言模型提供商"
    )
    
    # Embedding provider
    embedding_provider = st.sidebar.selectbox(
        "嵌入模型提供商",
        ["openai"],  # Only OpenAI embeddings for now
        help="选择文本嵌入模型"
    )
    
    # Advanced settings
    st.sidebar.subheader("高级设置")
    max_tokens = st.sidebar.slider("最大生成长度", 100, 4000, 2000)
    temperature = st.sidebar.slider("温度", 0.0, 1.0, 0.1)
    top_k_docs = st.sidebar.slider("检索文档数量", 1, 20, 5)
    
    return {
        "openai_api_key": openai_api_key,
        "google_api_key": google_api_key,
        "llm_provider": llm_provider,
        "embedding_provider": embedding_provider,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_k_docs": top_k_docs
    }

async def run_async_agentic_response(query: str, config: Dict[str, Any], collection_name: str = "tcm_tox", use_reasoning_tools: bool = True):
    """Wrapper to run async agentic response in Streamlit."""
    try:
        response = await create_agentic_response(
            query=query,
            config=config,
            collection_name=collection_name,
            use_reasoning_tools=use_reasoning_tools
        )
        return response
    except Exception as e:
        st.error(f"生成回答时出错: {str(e)}")
        return None

def ingest_section(config: Dict[str, Any]):
    """File upload and ingestion section."""
    st.subheader("📁 数据摄取")
    
    # File upload
    uploaded_file = st.file_uploader(
        "上传 Markdown 知识文件",
        type=['md', 'txt'],
        help="上传包含毒理学研究数据的 Markdown 文件"
    )
    
    if uploaded_file is not None:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as tmp_file:
            content = uploaded_file.read().decode('utf-8')
            tmp_file.write(content)
            tmp_file_path = tmp_file.name
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🚀 开始摄取", key="ingest_button"):
                try:
                    with st.spinner("正在处理文件..."):
                        # Call ingestion function (async)
                        result = asyncio.run(ingest_markdown_file(
                            file_path=tmp_file_path,
                            collection_name="tcm_tox"
                        ))
                        st.success(f"✅ 文件摄取成功！处理了 {result.get('chunks', 0)} 个文档块")
                except Exception as e:
                    st.error(f"❌ 摄取失败: {str(e)}")
                finally:
                    # Clean up temp file
                    if os.path.exists(tmp_file_path):
                        os.unlink(tmp_file_path)
        
        with col2:
            # Preview content
            st.text_area("文件预览", content[:500] + "..." if len(content) > 500 else content, height=100)

def gpt5_reasoning_tab(config: Dict[str, Any]):
    """GPT-5 retrieval and reasoning tab using new agentic pipeline."""
    st.header("🤖 检索与回答（GPT-5）")
    
    # Check API keys
    if not config["openai_api_key"] and not config["google_api_key"]:
        st.error("无法设置推理代理。请检查API密钥配置。")
        return
    
    # Query input
    query = st.text_area(
        "请输入您的毒理学问题:",
        placeholder="例如: 请分析某个TCM化合物对肝癌细胞的毒性作用机制...",
        height=100
    )
    
    col1, col2 = st.columns([3, 1])
    
    with col2:
        if st.button("🔍 分析", key="gpt5_analyze"):
            if not query.strip():
                st.warning("请输入查询内容")
                return
            
            with st.spinner("正在检索相关文档并分析..."):
                try:
                    # Use new agentic pipeline
                    response = asyncio.run(run_async_agentic_response(
                        query=query,
                        config=config,
                        collection_name="tcm_tox",
                        use_reasoning_tools=False  # Basic reasoning for this tab
                    ))
                    
                    if response and response.refusal_reason is None:
                        # Display results
                        st.subheader("📊 分析结果")
                        st.markdown(response.response_text)
                        
                        # Display evidence pack info
                        if response.evidence_pack.citation_ids:
                            st.subheader("📋 证据引用")
                            for citation in response.citations:
                                st.markdown(f"- {citation}")
                        
                        # Display retrieved sources
                        with st.expander("📚 参考文献来源"):
                            for i, doc in enumerate(response.evidence_pack.retrieved_docs, 1):
                                st.write(f"**来源 {i}:** {doc.get('document_title', '未知标题')}")
                                st.write(f"**节段:** {doc.get('section_type', '未知节段')}")
                                st.write(f"**摘要:** {doc.get('content', '')[:200]}...")
                                if doc.get('source_page'):
                                    st.write(f"**页面:** {doc.get('source_page')}")
                                st.write("---")
                        
                        # Display confidence and reasoning info
                        st.subheader("🔍 分析信息")
                        st.write(f"**置信度:** {response.confidence_score:.2f}")
                        st.write(f"**检索到的文档数:** {len(response.evidence_pack.retrieved_docs)}")
                        st.write(f"**引用数:** {len(response.citations)}")
                    
                    elif response and response.refusal_reason:
                        st.warning("⚠️ 分析受限")
                        st.markdown(response.response_text)
                        st.write(f"**原因:** {response.refusal_reason}")
                    
                    else:
                        st.error("分析失败，请重试")
                
                except Exception as e:
                    st.error(f"分析过程中出现错误: {str(e)}")
    
    with col1:
        # Show query parameters
        st.info(f"🔧 当前配置: {config['llm_provider'].upper()} | 检索文档数: {config['top_k_docs']} | 温度: {config['temperature']}")

def reasoning_visualization_tab(config: Dict[str, Any]):
    """Reasoning visualization tab with advanced agentic pipeline."""
    st.header("🧠 推理可视化（高级分析）")
    
    # Check API keys
    if not config["openai_api_key"] and not config["google_api_key"]:
        st.error("无法设置知识代理。请检查API密钥配置。")
        return
    
    # Query input
    query = st.text_area(
        "请输入复杂的毒理学研究问题:",
        placeholder="例如: 比较不同TCM化合物在肝癌治疗中的毒性差异，并提供实验设计方案...",
        height=100
    )
    
    col1, col2 = st.columns([3, 1])
    
    with col2:
        show_reasoning = st.checkbox("显示推理过程", value=True)
        if st.button("🧩 深度分析", key="reasoning_analyze"):
            if not query.strip():
                st.warning("请输入查询内容")
                return
            
            with st.spinner("正在进行深度推理分析..."):
                try:
                    # Use advanced agentic pipeline with reasoning tools
                    response = asyncio.run(run_async_agentic_response(
                        query=query,
                        config=config,
                        collection_name="tcm_tox",
                        use_reasoning_tools=True  # Advanced reasoning for this tab
                    ))
                    
                    if response and response.refusal_reason is None:
                        # Show reasoning steps if requested
                        if show_reasoning and response.reasoning_steps:
                            st.subheader("🔍 推理过程")
                            reasoning_container = st.container()
                            
                            with reasoning_container:
                                st.markdown("**推理步骤:**")
                                for i, step in enumerate(response.reasoning_steps, 1):
                                    st.markdown(f"{i}. {step}")
                        
                        # Display main results
                        st.subheader("📋 分析报告")
                        st.markdown(response.response_text)
                        
                        # Display evidence and citations
                        if response.evidence_pack.citation_ids:
                            st.subheader("📋 证据引用")
                            for citation in response.citations:
                                st.markdown(f"- {citation}")
                        
                        # Advanced analysis metrics
                        st.subheader("🔬 分析详情")
                        col_a, col_b, col_c = st.columns(3)
                        
                        with col_a:
                            st.metric("置信度", f"{response.confidence_score:.2f}")
                        
                        with col_b:
                            st.metric("证据文档", len(response.evidence_pack.retrieved_docs))
                        
                        with col_c:
                            st.metric("引用数量", len(response.citations))
                        
                        # Query decomposition info
                        st.subheader("🧩 查询分解")
                        st.write("**查询类型分类**: 自动识别并分解为多个子查询")
                        st.write("**证据检索**: 去重后的相关文档")
                        st.write("**推理合成**: 基于证据的结构化分析")
                        
                        # Display retrieved sources with advanced info
                        with st.expander("📚 详细文献来源"):
                            for i, doc in enumerate(response.evidence_pack.retrieved_docs, 1):
                                st.write(f"**文档 {i}:** {doc.get('document_title', '未知标题')}")
                                st.write(f"**节段类型:** {doc.get('section_type', '未知节段')}")
                                if doc.get('vector_score'):
                                    st.write(f"**相似度得分:** {doc.get('vector_score', 0):.3f}")
                                if doc.get('bm25_score'):
                                    st.write(f"**关键词得分:** {doc.get('bm25_score', 0):.3f}")
                                if doc.get('combined_score'):
                                    st.write(f"**综合得分:** {doc.get('combined_score', 0):.3f}")
                                st.write(f"**内容摘要:** {doc.get('content', '')[:200]}...")
                                if doc.get('source_page'):
                                    st.write(f"**源页面:** {doc.get('source_page')}")
                                st.write("---")
                    
                    elif response and response.refusal_reason:
                        st.warning("⚠️ 深度分析受限")
                        st.markdown(response.response_text)
                        st.write(f"**限制原因:** {response.refusal_reason}")
                    
                    else:
                        st.error("深度分析失败，请重试")
                
                except Exception as e:
                    st.error(f"推理分析过程中出现错误: {str(e)}")
    
    with col1:
        # Advanced reasoning parameters
        st.info(f"🔧 高级配置: {config['llm_provider'].upper()} + 知识推理工具 | 嵌入: text-embedding-3-large")
        
        # Show reasoning tools info
        with st.expander("ℹ️ 高级推理功能"):
            st.markdown("""
            **智能查询分解:**
            - 🧠 **自动分类**: 机制/毒性/设计/对比/一般
            - 🔍 **多角度检索**: 子查询并行搜索
            - 📊 **证据聚合**: 去重和相关性排序
            
            **推理增强功能:**
            - 🤔 **结构化思考**: 分步分析问题
            - 🔍 **知识库搜索**: 混合检索策略
            - 📊 **证据评估**: 置信度计算
            - 🛡️ **安全防护**: 拒绝回答不充分问题
            
            **适用复杂场景:**
            - 多化合物对比分析
            - 实验设计方案制定
            - 机制研究综合评估
            - 安全性风险分析
            """)

def main():
    """Main application entry point."""
    setup_page_config()
    
    # Header
    st.title("🧬 ToxiRAG - 肝癌毒性预测系统")
    st.markdown("*基于检索增强生成的AI辅助动物实验预测平台*")
    
    # Sidebar configuration
    config = setup_sidebar()
    
    # Check API keys
    if not config["openai_api_key"] and not config["google_api_key"]:
        st.warning("⚠️ 请在侧边栏配置至少一个API密钥以继续使用")
        return
    
    # File ingestion section
    with st.expander("📁 知识库管理", expanded=False):
        ingest_section(config)
    
    # Main tabs
    tab1, tab2 = st.tabs(["🤖 检索与回答（GPT-5）", "🧠 推理可视化（OpenAI embed）"])
    
    with tab1:
        gpt5_reasoning_tab(config)
    
    with tab2:
        reasoning_visualization_tab(config)
    
    # Footer
    st.markdown("---")
    st.markdown(
        "💡 **提示**: 使用推理可视化功能可以看到AI的思考过程，"
        "适合复杂的毒理学分析和实验设计。"
    )

if __name__ == "__main__":
    main() 