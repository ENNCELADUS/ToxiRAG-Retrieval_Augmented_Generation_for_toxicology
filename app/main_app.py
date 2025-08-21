"""
ToxiRAG Streamlit Application
Main interface for liver cancer toxicity prediction and experiment planning.
"""

import os
import streamlit as st
import tempfile
from pathlib import Path
from typing import Optional, List, Dict, Any

# Agno imports for reasoning tools
from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.models.google import Gemini
from agno.tools.reasoning import ReasoningTools
from agno.tools.knowledge import KnowledgeTools
from agno.embedder.openai import OpenAIEmbedder
from agno.knowledge.text import TextKnowledgeBase
from agno.vectordb.lancedb import LanceDb, SearchType

# Local imports (will be created)
try:
    from ingest.ingest_local import ingest_markdown_file, setup_qdrant_collection
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

def setup_agent(config: Dict[str, Any], agent_type: str = "reasoning") -> Optional[Agent]:
    """Setup Agno agent with reasoning tools."""
    try:
        # Set environment variables
        if config["openai_api_key"]:
            os.environ["OPENAI_API_KEY"] = config["openai_api_key"]
        if config["google_api_key"]:
            os.environ["GOOGLE_API_KEY"] = config["google_api_key"]
        
        # Select model based on provider
        if config["llm_provider"] == "openai":
            model = OpenAIChat(
                id="gpt-5-nano", 
                api_key=config["openai_api_key"],
                max_tokens=config["max_tokens"],
                temperature=config["temperature"]
            )
        else:  # gemini
            model = Gemini(
                id="gemini-2.5-flash",
                api_key=config["google_api_key"],
                max_tokens=config["max_tokens"],
                temperature=config["temperature"]
            )
        
        # Setup tools based on agent type
        if agent_type == "reasoning":
            tools = [ReasoningTools(add_instructions=True)]
        elif agent_type == "knowledge":
            # Setup knowledge base for toxicology data
            knowledge = TextKnowledgeBase(
                vector_db=LanceDb(
                    uri="tmp/toxirag_lancedb",
                    table_name="toxicology_docs",
                    search_type=SearchType.hybrid,
                    embedder=OpenAIEmbedder(
                        id="text-embedding-3-large",
                        api_key=config["openai_api_key"]
                    )
                )
            )
            tools = [KnowledgeTools(
                knowledge=knowledge,
                think=True,
                search=True,
                analyze=True,
                add_few_shot=True
            )]
        else:
            tools = []
        
        agent = Agent(
            model=model,
            tools=tools,
            instructions="""You are a toxicology expert specializing in liver cancer research and Traditional Chinese Medicine (TCM) compounds. 
            Provide detailed, evidence-based responses about toxicity predictions and experimental planning.
            Always cite your sources and explain your reasoning step by step.""",
            show_tool_calls=True,
            markdown=True
        )
        
        return agent
    except Exception as e:
        st.error(f"设置代理时出错: {str(e)}")
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
                        # Call ingestion function
                        result = ingest_markdown_file(
                            file_path=tmp_file_path,
                            collection_name="tcm_tox"
                        )
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
    """GPT-5 retrieval and reasoning tab."""
    st.header("🤖 检索与回答（GPT-5）")
    
    # Setup reasoning agent
    agent = setup_agent(config, "reasoning")
    if not agent:
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
                    # Step 1: Retrieve relevant documents
                    retrieved_docs = retrieve_relevant_docs(
                        query=query,
                        top_k=config["top_k_docs"],
                        collection_name="tcm_tox"
                    )
                    
                    # Step 2: Create context-aware query
                    context = "\n\n".join([doc.get('content', '') for doc in retrieved_docs])
                    enhanced_query = f"""
                    基于以下检索到的毒理学文献上下文，请回答用户的问题:
                    
                    上下文文档:
                    {context}
                    
                    用户问题: {query}
                    
                    请提供详细的分析，包括:
                    1. 相关的毒理学机制
                    2. 实验设计建议
                    3. 安全性考虑
                    4. 引用的证据来源
                    """
                    
                    # Step 3: Generate reasoning response
                    response = agent.run(enhanced_query)
                    
                    # Display results
                    st.subheader("📊 分析结果")
                    st.markdown(response.content)
                    
                    # Display retrieved sources
                    with st.expander("📚 参考文献来源"):
                        for i, doc in enumerate(retrieved_docs, 1):
                            st.write(f"**来源 {i}:** {doc.get('title', '未知标题')}")
                            st.write(f"**摘要:** {doc.get('content', '')[:200]}...")
                            st.write("---")
                
                except Exception as e:
                    st.error(f"分析过程中出现错误: {str(e)}")
    
    with col1:
        # Show query parameters
        st.info(f"🔧 当前配置: {config['llm_provider'].upper()} | 检索文档数: {config['top_k_docs']} | 温度: {config['temperature']}")

def reasoning_visualization_tab(config: Dict[str, Any]):
    """Reasoning visualization tab with OpenAI embeddings."""
    st.header("🧠 推理可视化（OpenAI 嵌入）")
    
    # Setup knowledge agent
    agent = setup_agent(config, "knowledge")
    if not agent:
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
                    # Enhanced query for reasoning visualization
                    enhanced_query = f"""
                    作为毒理学专家，请对以下问题进行深度分析:
                    
                    {query}
                    
                    请按照以下步骤进行分析:
                    1. 首先思考问题的关键要素
                    2. 搜索相关的毒理学知识
                    3. 分析不同化合物的作用机制
                    4. 比较安全性和有效性
                    5. 提供具体的实验设计建议
                    6. 总结关键发现和建议
                    """
                    
                    # Generate response with reasoning tools
                    if show_reasoning:
                        # Show intermediate reasoning steps
                        st.subheader("🔍 推理过程")
                        reasoning_container = st.container()
                        
                        response = agent.run(
                            enhanced_query,
                            stream=True,
                            show_full_reasoning=True
                        )
                        
                        with reasoning_container:
                            st.markdown("**推理步骤:**")
                            # Note: Actual reasoning visualization would need custom streaming handler
                            st.info("推理工具正在分析问题...")
                    else:
                        response = agent.run(enhanced_query)
                    
                    # Display final results
                    st.subheader("📋 分析报告")
                    st.markdown(response.content)
                    
                    # Additional insights
                    st.subheader("💡 关键洞察")
                    insights_query = f"基于上述分析，请总结3个最重要的毒理学洞察和实验建议: {query}"
                    insights_response = agent.run(insights_query)
                    st.markdown(insights_response.content)
                
                except Exception as e:
                    st.error(f"推理分析过程中出现错误: {str(e)}")
    
    with col1:
        # Reasoning parameters
        st.info(f"🔧 推理配置: 知识搜索 + 分析工具 | 嵌入模型: text-embedding-3-large")
        
        # Show reasoning tools info
        with st.expander("ℹ️ 推理工具说明"):
            st.markdown("""
            **推理工具功能:**
            - 🤔 **Think**: 结构化思考空间
            - 🔍 **Search**: 知识库搜索
            - 📊 **Analyze**: 结果分析工具
            
            **适用场景:**
            - 复杂的多步骤分析
            - 需要对比多个化合物
            - 实验设计方案制定
            - 安全性评估
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