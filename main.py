import streamlit as st
import pandas as pd
import openai
from openai import OpenAI
import json
import re
from datetime import datetime
from typing import List, Dict, Any, Optional
import warnings
import io
import sys
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="CSV Chat Assistant",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

class StreamlitCSVChatAssistant:
    def __init__(self, api_key: str, dataframe: pd.DataFrame, column_metadata: Dict[str, str]):
        """
        Initialize the CSV Chat Assistant for Streamlit
        
        Args:
            api_key: OpenAI API key
            dataframe: Loaded pandas DataFrame
            column_metadata: Dictionary mapping column names to their descriptions/types
        """
        self.client = OpenAI(api_key=api_key)
        self.df = dataframe
        self.column_metadata = column_metadata
        
        # Initialize session state for conversation history
        if 'conversation_history' not in st.session_state:
            st.session_state.conversation_history = []
    
    def get_system_message(self) -> str:
        """Generate system message with current dataframe columns and metadata"""
        columns_info = []
        for col in self.df.columns:
            dtype = str(self.df[col].dtype)
            metadata_info = self.column_metadata.get(col, "No description provided")
            columns_info.append(f"- {col}: {dtype} ({metadata_info})")
        
        columns_text = "\n".join(columns_info)
        
        # Auto-detect datetime columns
        datetime_columns = [col for col in self.df.columns if pd.api.types.is_datetime64_any_dtype(self.df[col])]
        datetime_info = f"\nDatetime columns for time-based analysis: {datetime_columns}" if datetime_columns else ""
        
        return f"""
You are a data analysis assistant specialized in working with uploaded CSV data.
Your job is to take a user's natural language question and generate the correct **pandas expression/code** 
that executes against the dataframe `df`.

CRITICAL RULES:
- Generate ONLY executable pandas code - no explanations, no markdown, no comments
- Always use pandas syntax only
- Do not hallucinate columns; use only those listed below
- Handle datetime columns carefully using .dt accessor when needed
- Only return aggregation, filtering, or summary results - NEVER return full raw data
- If user asks for all data, refuse and suggest using .head() or summary methods
- For irrelevant questions, return: print("I can only help with data analysis questions about the uploaded CSV file.")

Available columns and metadata:
{columns_text}{datetime_info}

IMPORTANT CONSIDERATIONS:
- For time-based queries, use .dt accessor for datetime operations
- Always limit results to reasonable sizes (use .head(), .tail(), or aggregations)
- Handle missing values appropriately with .fillna() or .dropna()
- Use proper pandas methods for grouping, filtering, and aggregation
- For large results, use .head(20) to limit output

OUTPUT REQUIREMENTS:
- Return ONLY the pandas code that can be executed
- No explanations, no markdown formatting, no comments
- Code should be ready to execute with exec()
"""

    def generate_pandas_code(self, user_question: str) -> str:
        """Generate pandas code based on user question and conversation context"""
        
        # Prepare context from conversation history
        context_messages = []
        if st.session_state.conversation_history:
            recent_history = st.session_state.conversation_history[-6:]  # Last 3 exchanges
            for entry in recent_history:
                context_messages.append(f"User: {entry['user_question']}")
                if entry.get('result_summary'):
                    context_messages.append(f"Result: {entry['result_summary']}")
        
        context_text = "\n".join(context_messages) if context_messages else "No previous context"
        
        prompt = f"""
{self.get_system_message()}

CONVERSATION CONTEXT:
{context_text}

CURRENT USER QUESTION: {user_question}

Generate the pandas code to answer this question:
"""
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=500
            )
            
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"print('Error generating code: {e}')"

    def execute_pandas_code(self, code: str) -> Dict[str, Any]:
        """Execute pandas code and return structured results"""
        try:
            # Clean the code
            code = code.strip()
            if code.startswith('```'):
                code = re.sub(r'^```[^\n]*\n', '', code)
                code = re.sub(r'\n```$', '', code)
            
            # Create safe execution environment
            safe_globals = {
                'df': self.df,
                'pd': pd,
                'datetime': datetime,
                '__builtins__': {
                    'print': print, 'len': len, 'str': str, 'int': int,
                    'float': float, 'round': round, 'sum': sum, 'min': min, 'max': max
                }
            }
            
            # Capture output
            old_stdout = sys.stdout
            sys.stdout = captured_output = io.StringIO()
            
            result = None
            try:
                exec(code, safe_globals)
                
                # If no printed output, try evaluating as expression
                if not captured_output.getvalue().strip():
                    result = eval(code, safe_globals)
            finally:
                sys.stdout = old_stdout
            
            output = captured_output.getvalue().strip()
            
            return {
                'success': True,
                'result': result,
                'output': output,
                'code': code,
                'error': None
            }
            
        except Exception as e:
            return {
                'success': False,
                'result': None,
                'output': '',
                'code': code,
                'error': str(e)
            }

    def process_question(self, user_question: str) -> Dict[str, Any]:
        """Process user question and return structured result for Streamlit"""
        if not user_question.strip():
            return {
                'success': False,
                'message': "Please ask a question about your data.",
                'result_type': 'error'
            }
        
        # Check for irrelevant questions
        irrelevant_keywords = ['joke', 'weather', 'news', 'hello', 'how are you', 'what is your name']
        if any(keyword in user_question.lower() for keyword in irrelevant_keywords):
            if not any(data_keyword in user_question.lower() for data_keyword in ['data', 'sales', 'revenue', 'customer', 'product']):
                return {
                    'success': False,
                    'message': "I can only help with data analysis questions about your uploaded CSV file.",
                    'result_type': 'error'
                }
        
        # Generate pandas code
        pandas_code = self.generate_pandas_code(user_question)
        
        # Execute the code
        execution_result = self.execute_pandas_code(pandas_code)
        
        # Process results for Streamlit display
        if execution_result['success']:
            result = execution_result['result']
            output = execution_result['output']
            
            # Determine result type and format
            if output:
                display_result = {
                    'success': True,
                    'result_type': 'text',
                    'content': output,
                    'code': pandas_code
                }
            elif isinstance(result, pd.DataFrame):
                display_result = {
                    'success': True,
                    'result_type': 'dataframe',
                    'content': result,
                    'code': pandas_code,
                    'row_count': len(result)
                }
            elif isinstance(result, pd.Series):
                display_result = {
                    'success': True,
                    'result_type': 'series',
                    'content': result,
                    'code': pandas_code
                }
            elif isinstance(result, (int, float)):
                display_result = {
                    'success': True,
                    'result_type': 'numeric',
                    'content': result,
                    'code': pandas_code
                }
            else:
                display_result = {
                    'success': True,
                    'result_type': 'other',
                    'content': str(result),
                    'code': pandas_code
                }
        else:
            display_result = {
                'success': False,
                'result_type': 'error',
                'content': f"Error: {execution_result['error']}",
                'code': pandas_code
            }
        
        # Add to conversation history
        self.add_to_conversation_history(user_question, display_result)
        
        return display_result

    def add_to_conversation_history(self, user_question: str, result: Dict[str, Any]):
        """Add interaction to conversation history"""
        # Create summary for context
        if result['success']:
            if result['result_type'] == 'dataframe':
                summary = f"Returned DataFrame with {result.get('row_count', 0)} rows"
            elif result['result_type'] == 'series':
                summary = f"Returned Series with {len(result['content']) if hasattr(result['content'], '__len__') else 'N/A'} values"
            elif result['result_type'] == 'numeric':
                summary = f"Returned numeric value: {result['content']}"
            else:
                summary = "Query executed successfully"
        else:
            summary = f"Error occurred"
        
        st.session_state.conversation_history.append({
            'user_question': user_question,
            'result': result,
            'result_summary': summary,
            'timestamp': datetime.now().isoformat()
        })
        
        # Keep only last 20 interactions
        if len(st.session_state.conversation_history) > 20:
            st.session_state.conversation_history = st.session_state.conversation_history[-20:]

def display_result(result: Dict[str, Any]):
    """Display results in Streamlit with proper formatting"""
    if not result['success']:
        st.error(f"❌ {result['content']}")
        if 'code' in result:
            with st.expander("Generated Code"):
                st.code(result['code'], language='python')
        return
    
    # Show generated code in expander
    with st.expander("🔧 Generated Pandas Code"):
        st.code(result['code'], language='python')
    
    # Display result based on type
    if result['result_type'] == 'dataframe':
        df_result = result['content']
        if len(df_result) == 0:
            st.info("📊 No data found matching your criteria.")
        else:
            st.success(f"📊 **Results** ({len(df_result)} rows)")
            
            # Display as Streamlit dataframe for better interactivity
            st.dataframe(
                df_result,
                use_container_width=True,
                hide_index=False
            )
            
            # Option to download results
            csv = df_result.to_csv(index=True)
            st.download_button(
                label="📥 Download Results as CSV",
                data=csv,
                file_name=f"query_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
    
    elif result['result_type'] == 'series':
        series_result = result['content']
        st.success("📊 **Results**")
        
        # Convert series to dataframe for better display
        df_display = series_result.reset_index()
        st.dataframe(df_display, use_container_width=True, hide_index=True)
    
    elif result['result_type'] == 'numeric':
        numeric_result = result['content']
        if isinstance(numeric_result, float):
            st.success(f"🔢 **Result**: {numeric_result:,.2f}")
        else:
            st.success(f"🔢 **Result**: {numeric_result:,}")
    
    elif result['result_type'] == 'text':
        st.success("✅ **Result**")
        st.text(result['content'])
    
    else:
        st.success(f"✅ **Result**: {result['content']}")

def create_column_metadata_form(df: pd.DataFrame) -> Dict[str, str]:
    """Create form for users to input column metadata"""
    st.subheader("📝 Column Metadata Configuration")
    st.write("Provide descriptions for your columns to help the assistant understand your data better.")
    
    metadata = {}
    
    # Create columns for better layout
    col1, col2 = st.columns(2)
    
    for i, column in enumerate(df.columns):
        with col1 if i % 2 == 0 else col2:
            # Auto-suggest based on column name and type
            dtype = str(df[column].dtype)
            auto_suggestion = get_auto_suggestion(column, dtype)
            
            metadata[column] = st.text_input(
                f"**{column}** ({dtype})",
                value=auto_suggestion,
                key=f"metadata_{column}",
                help=f"Describe what this column represents"
            )
    
    return metadata

def get_auto_suggestion(column_name: str, dtype: str) -> str:
    """Auto-suggest column descriptions based on name and type"""
    column_lower = column_name.lower()
    
    # Common patterns
    if 'id' in column_lower:
        return "Unique identifier"
    elif 'date' in column_lower or 'time' in column_lower:
        return "Date/time information"
    elif 'amount' in column_lower or 'price' in column_lower or 'cost' in column_lower:
        return "Monetary value"
    elif 'quantity' in column_lower or 'qty' in column_lower:
        return "Quantity/count"
    elif 'name' in column_lower:
        return "Name/description"
    elif 'category' in column_lower:
        return "Category/classification"
    elif 'branch' in column_lower:
        return "Branch/location information"
    elif 'discount' in column_lower:
        return "Discount amount/percentage"
    elif 'revenue' in column_lower or 'sales' in column_lower:
        return "Sales/revenue amount"
    elif dtype.startswith('int') or dtype.startswith('float'):
        return "Numeric value"
    elif dtype == 'object':
        return "Text/categorical data"
    else:
        return "Data field"

def initialize_session_state():
    """Initialize Streamlit session state variables"""
    if 'assistant' not in st.session_state:
        st.session_state.assistant = None
    if 'df' not in st.session_state:
        st.session_state.df = None
    if 'column_metadata' not in st.session_state:
        st.session_state.column_metadata = {}
    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False

def main():
    """Main Streamlit application"""
    initialize_session_state()
    
    # Header
    st.title("📊 CSV Chat Assistant")
    st.markdown("Upload your CSV file and ask questions about your data in natural language!")
    
    # Sidebar for file upload and configuration
    with st.sidebar:
        st.header("🔧 Setup")
        
        # API Key input
        api_key = st.text_input(
            "OpenAI API Key",
            type="password",
            help="Enter your OpenAI API key"
        )
        
        if not api_key:
            st.warning("⚠️ Please enter your OpenAI API key to continue")
            return
        
        # File upload
        st.subheader("📁 Upload CSV File")
        uploaded_file = st.file_uploader(
            "Choose a CSV file",
            type=['csv'],
            help="Upload your CSV file for analysis"
        )
        
        if uploaded_file is not None:
            try:
                # Load the dataframe
                df = pd.read_csv(uploaded_file)
                st.session_state.df = df
                
                # Auto-convert potential datetime columns
                for col in df.columns:
                    if 'date' in col.lower() and df[col].dtype == 'object':
                        try:
                            df[col] = pd.to_datetime(df[col], errors='coerce')
                            st.session_state.df = df
                        except:
                            pass
                
                st.success(f"✅ File loaded: {len(df)} rows, {len(df.columns)} columns")
                
                # Show basic info
                with st.expander("📋 Data Overview"):
                    st.write(f"**Rows**: {len(df):,}")
                    st.write(f"**Columns**: {len(df.columns)}")
                    st.write("**Column Types**:")
                    for col in df.columns:
                        st.write(f"- {col}: {df[col].dtype}")
                
                # Show sample data
                with st.expander("👀 Sample Data"):
                    st.dataframe(df.head(), use_container_width=True)
                
                st.session_state.data_loaded = True
                
            except Exception as e:
                st.error(f"❌ Error loading file: {e}")
                return
    
    # Main content area
    if not st.session_state.data_loaded:
        st.info("👆 Please upload a CSV file in the sidebar to get started!")
        return
    
    # Column metadata configuration
    if st.session_state.df is not None:
        with st.expander("📝 Configure Column Metadata (Optional but Recommended)", expanded=False):
            st.write("Help the assistant understand your data better by providing column descriptions:")
            
            metadata = create_column_metadata_form(st.session_state.df)
            
            if st.button("💾 Save Metadata Configuration"):
                st.session_state.column_metadata = metadata
                st.success("✅ Metadata saved!")
                
                # Initialize assistant with metadata
                st.session_state.assistant = StreamlitCSVChatAssistant(
                    api_key=api_key,
                    dataframe=st.session_state.df,
                    column_metadata=metadata
                )
                st.rerun()
    
    # Initialize assistant if not done yet
    if st.session_state.assistant is None and st.session_state.df is not None:
        st.session_state.assistant = StreamlitCSVChatAssistant(
            api_key=api_key,
            dataframe=st.session_state.df,
            column_metadata=st.session_state.column_metadata
        )
    
    # Chat interface
    st.header("💬 Chat with Your Data")
    
    # Display conversation history
    if st.session_state.conversation_history:
        st.subheader("📜 Conversation History")
        
        for i, entry in enumerate(reversed(st.session_state.conversation_history[-5:])):
            with st.container():
                st.markdown(f"**❓ Question {len(st.session_state.conversation_history) - i}:** {entry['user_question']}")
                
                if entry['result']['success']:
                    display_result(entry['result'])
                else:
                    st.error(entry['result']['content'])
                
                st.markdown("---")
    
    # Question input
    st.subheader("🤔 Ask a Question")
    
    # Example questions
    with st.expander("💡 Example Questions"):
        example_questions = [
            "What's the total sales amount?",
            "Show me the top 10 rows by revenue",
            "What's the average order value?",
            "Group sales by category and show totals",
            "Show me monthly sales trends",
            "What are the unique values in the category column?",
            "Filter data for sales above 1000",
            "Show summary statistics for numeric columns"
        ]
        
        for question in example_questions:
            if st.button(f"📝 {question}", key=f"example_{question}"):
                st.session_state.current_question = question
    
    # Question input methods
    tab1, tab2 = st.tabs(["💬 Type Question", "🎤 Quick Questions"])
    
    with tab1:
        user_question = st.text_input(
            "Enter your question about the data:",
            value=st.session_state.get('current_question', ''),
            placeholder="e.g., What's the total sales by category?"
        )
        
        col1, col2, col3 = st.columns([1, 1, 2])
        with col1:
            ask_button = st.button("🚀 Ask Question", type="primary")
        with col2:
            clear_button = st.button("🗑️ Clear Chat")
        
        if clear_button:
            st.session_state.conversation_history = []
            st.session_state.current_question = ""
            st.rerun()
    
    with tab2:
        st.write("Quick analysis options:")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📊 Data Summary"):
                user_question = "Show me summary statistics for all numeric columns"
                ask_button = True
            if st.button("🔝 Top 10 Rows"):
                user_question = "Show me the first 10 rows"
                ask_button = True
        
        with col2:
            if st.button("📈 Column Info"):
                user_question = "Show me information about all columns"
                ask_button = True
            if st.button("🔍 Data Types"):
                user_question = "Show me the data types of all columns"
                ask_button = True
        
        with col3:
            if st.button("❓ Missing Values"):
                user_question = "Show me missing values count for each column"
                ask_button = True
            if st.button("📏 Shape Info"):
                user_question = "What's the shape of the dataset?"
                ask_button = True
    
    # Process question
    if ask_button and user_question and st.session_state.assistant:
        with st.spinner("🔄 Analyzing your question..."):
            result = st.session_state.assistant.process_question(user_question)
        
        st.subheader(f"📋 Answer to: \"{user_question}\"")
        display_result(result)
        
        # Clear the current question
        if 'current_question' in st.session_state:
            del st.session_state.current_question
    
    # Footer with tips
    st.markdown("---")
    with st.expander("💡 Tips for Better Results"):
        st.markdown("""
        **🎯 For Best Results:**
        - Be specific about what you want to analyze
        - Use column names that exist in your data
        - Ask for summaries, trends, or specific filtering
        - Avoid asking for complete datasets (use "top 10" instead)
        
        **📊 Good Questions:**
        - "What's the total revenue by category?"
        - "Show me the top 5 customers by purchase amount"
        - "What's the sales trend over time?"
        - "Filter products with price above 100"
        
        **❌ Avoid:**
        - "Show me all data" (too large)
        - Questions unrelated to your data
        - Asking for columns that don't exist
        """)

# Additional utility functions for enhanced features
def create_data_profiling_report(df: pd.DataFrame):
    """Create a data profiling report"""
    st.subheader("📊 Data Profiling Report")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Rows", f"{len(df):,}")
    with col2:
        st.metric("Total Columns", len(df.columns))
    with col3:
        st.metric("Memory Usage", f"{df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
    with col4:
        st.metric("Missing Values", f"{df.isnull().sum().sum():,}")
    
    # Column analysis
    st.subheader("🔍 Column Analysis")
    
    analysis_data = []
    for col in df.columns:
        analysis_data.append({
            'Column': col,
            'Type': str(df[col].dtype),
            'Non-Null Count': f"{df[col].count():,}",
            'Null Count': f"{df[col].isnull().sum():,}",
            'Unique Values': f"{df[col].nunique():,}",
            'Sample Value': str(df[col].iloc[0]) if len(df) > 0 else "N/A"
        })
    
    analysis_df = pd.DataFrame(analysis_data)
    st.dataframe(analysis_df, use_container_width=True, hide_index=True)

def sidebar_analytics(df: pd.DataFrame):
    """Add analytics sidebar"""
    with st.sidebar:
        st.header("📈 Quick Analytics")
        
        # Numeric columns analysis
        numeric_cols = df.select_dtypes(include=['number']).columns
        if len(numeric_cols) > 0:
            st.subheader("🔢 Numeric Columns")
            selected_numeric = st.selectbox(
                "Select column for quick stats:",
                numeric_cols,
                key="numeric_analysis"
            )
            
            if selected_numeric:
                col_data = df[selected_numeric].dropna()
                st.write(f"**{selected_numeric} Statistics:**")
                st.write(f"- Mean: {col_data.mean():.2f}")
                st.write(f"- Median: {col_data.median():.2f}")
                st.write(f"- Min: {col_data.min():.2f}")
                st.write(f"- Max: {col_data.max():.2f}")
        
        # Categorical columns analysis  
        categorical_cols = df.select_dtypes(include=['object']).columns
        if len(categorical_cols) > 0:
            st.subheader("📝 Categorical Columns")
            selected_categorical = st.selectbox(
                "Select column for value counts:",
                categorical_cols,
                key="categorical_analysis"
            )
            
            if selected_categorical:
                value_counts = df[selected_categorical].value_counts().head()
                st.write(f"**Top values in {selected_categorical}:**")
                for value, count in value_counts.items():
                    st.write(f"- {value}: {count}")

# Enhanced main function with additional features
def enhanced_main():
    """Enhanced main function with additional features"""
    main()
    
    # Add data profiling option
    if st.session_state.data_loaded and st.session_state.df is not None:
        # Add analytics sidebar
        sidebar_analytics(st.session_state.df)
        
        # Add data profiling option
        if st.checkbox("📊 Show Detailed Data Profiling"):
            create_data_profiling_report(st.session_state.df)

if __name__ == "__main__":
    enhanced_main()

