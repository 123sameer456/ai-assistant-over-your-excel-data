import pandas as pd
import openai
from openai import OpenAI
import json
import re
from datetime import datetime
from typing import List, Dict, Any
import warnings
warnings.filterwarnings('ignore')

class CSVChatAssistant:
    def __init__(self, api_key: str, csv_file_path: str):
        """
        Initialize the CSV Chat Assistant
        
        Args:
            api_key: OpenAI API key
            csv_file_path: Path to the CSV file
        """
        self.client = OpenAI(api_key=api_key)
        self.csv_file_path = csv_file_path
        self.df = None
        self.conversation_history = []
        self.load_data()
        
    def load_data(self):
        """Load and prepare the CSV data"""
        try:
            self.df = pd.read_csv(self.csv_file_path)
            # Convert InvoiceDate to datetime if it exists
            if 'InvoiceDate' in self.df.columns:
                self.df['InvoiceDate'] = pd.to_datetime(self.df['InvoiceDate'], errors='coerce')
            print(f"✅ Data loaded successfully: {len(self.df)} rows, {len(self.df.columns)} columns")
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            
    def get_system_message(self) -> str:
        """Get the system message with column information"""
        columns_info = []
        for col in self.df.columns:
            dtype = str(self.df[col].dtype)
            columns_info.append(f"- {col}: {dtype}")
        
        columns_text = "\n".join(columns_info)
        
        return f"""
You are a data analysis assistant specialized in working with CSV data.
Your job is to take a user's natural language question and generate the correct **pandas expression/code** 
that executes against the dataframe `df`. 

CRITICAL RULES:
- Generate ONLY executable pandas code - no explanations, no markdown, no comments
- Always use pandas syntax only
- Do not hallucinate columns; use only those listed below
- Handle the `InvoiceDate` column carefully (datetime format)
- Only return aggregation, filtering, or summary results - NEVER return full raw data
- If user asks for all data, refuse and suggest using .head() or summary methods
- For irrelevant questions, return: print("I can only help with data analysis questions about the CSV file.")

Available columns:
{columns_text}

IMPORTANT CONSIDERATIONS:
- For time-based queries, use .dt accessor for datetime operations
- Always limit results to reasonable sizes (use .head(), .tail(), or aggregations)
- Handle missing values appropriately
- Use proper pandas methods for grouping, filtering, and aggregation

OUTPUT REQUIREMENTS:
- Return ONLY the pandas code that can be executed
- No explanations, no markdown formatting, no comments
- Code should be ready to execute with exec()
"""

    def generate_pandas_code(self, user_question: str, context: str = "") -> str:
        """Generate pandas code based on user question and conversation context"""
        
        # Prepare context from conversation history
        context_messages = []
        if self.conversation_history:
            recent_history = self.conversation_history[-6:]  # Last 3 exchanges
            for entry in recent_history:
                context_messages.append(f"User: {entry['user_question']}")
                if entry['result_summary']:
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
        """
        Safely execute pandas code and return results
        
        Returns:
            Dict containing success status, result, and error info
        """
        try:
            # Clean the code
            code = code.strip()
            # Remove markdown code blocks if present
            if code.startswith('```'):
                code = re.sub(r'^```[^\n]*\n', '', code)
                code = re.sub(r'\n```$', '', code)
            
            # Create safe execution environment
            safe_globals = {
                'df': self.df,
                'pd': pd,
                'datetime': datetime,
                '__builtins__': {
                    'print': print,
                    'len': len,
                    'str': str,
                    'int': int,
                    'float': float,
                    'round': round,
                    'sum': sum,
                    'min': min,
                    'max': max,
                }
            }
            
            # Capture output
            import io
            import sys
            
            old_stdout = sys.stdout
            sys.stdout = captured_output = io.StringIO()
            
            result = None
            try:
                # Execute the code
                exec(code, safe_globals)
                
                # If code doesn't print anything, try to evaluate it as expression
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

    def format_result(self, execution_result: Dict[str, Any]) -> str:
        """Format the execution result for display"""
        if not execution_result['success']:
            return f"❌ Error executing query: {execution_result['error']}"
        
        result = execution_result['result']
        output = execution_result['output']
        
        # If there's printed output, use that
        if output:
            return output
        
        # Format different types of results
        if result is None:
            return "✅ Query executed successfully (no output)"
        
        if isinstance(result, (pd.DataFrame, pd.Series)):
            if isinstance(result, pd.DataFrame):
                if len(result) == 0:
                    return "📊 No data found matching your criteria."
                
                # Format as nice table
                if len(result) <= 20:  # Show full table for small results
                    return f"📊 Results ({len(result)} rows):\n\n{result.to_string(index=True)}"
                else:
                    return f"📊 Results (showing first 20 of {len(result)} rows):\n\n{result.head(20).to_string(index=True)}"
            
            elif isinstance(result, pd.Series):
                if len(result) <= 10:
                    return f"📊 Results:\n\n{result.to_string()}"
                else:
                    return f"📊 Results (showing first 10 of {len(result)} items):\n\n{result.head(10).to_string()}"
        
        elif isinstance(result, (int, float)):
            if isinstance(result, float):
                return f"🔢 Result: {result:,.2f}"
            else:
                return f"🔢 Result: {result:,}"
        
        elif isinstance(result, str):
            return f"📝 Result: {result}"
        
        else:
            return f"✅ Result: {str(result)}"

    def add_to_conversation_history(self, user_question: str, execution_result: Dict[str, Any], formatted_result: str):
        """Add interaction to conversation history"""
        # Create summary for context
        if execution_result['success']:
            if isinstance(execution_result['result'], pd.DataFrame):
                summary = f"Returned DataFrame with {len(execution_result['result'])} rows"
            elif isinstance(execution_result['result'], pd.Series):
                summary = f"Returned Series with {len(execution_result['result'])} values"
            elif isinstance(execution_result['result'], (int, float)):
                summary = f"Returned numeric value: {execution_result['result']}"
            else:
                summary = "Query executed successfully"
        else:
            summary = f"Error: {execution_result['error']}"
        
        self.conversation_history.append({
            'user_question': user_question,
            'generated_code': execution_result['code'],
            'result_summary': summary,
            'timestamp': datetime.now().isoformat()
        })
        
        # Keep only last 20 interactions to manage memory
        if len(self.conversation_history) > 20:
            self.conversation_history = self.conversation_history[-20:]

    def process_question(self, user_question: str) -> str:
        """
        Process a user question and return formatted result
        
        Args:
            user_question: Natural language question about the data
            
        Returns:
            Formatted result string
        """
        if not user_question.strip():
            return "❓ Please ask a question about your data."
        
        # Check for irrelevant questions
        irrelevant_keywords = ['joke', 'weather', 'news', 'hello', 'how are you', 'what is your name']
        if any(keyword in user_question.lower() for keyword in irrelevant_keywords):
            if not any(data_keyword in user_question.lower() for data_keyword in ['data', 'sales', 'revenue', 'customer', 'product']):
                return "🤖 I can only help with data analysis questions about your CSV file. Please ask about your sales data!"
        
        print(f"🤔 Processing question: {user_question}")
        
        # Generate pandas code
        pandas_code = self.generate_pandas_code(user_question)
        print(f"🔧 Generated code: {pandas_code}")
        
        # Execute the code
        execution_result = self.execute_pandas_code(pandas_code)
        
        # Format the result
        formatted_result = self.format_result(execution_result)
        
        # Add to conversation history
        self.add_to_conversation_history(user_question, execution_result, formatted_result)
        
        return formatted_result

    def get_data_overview(self) -> str:
        """Get an overview of the loaded data"""
        if self.df is None:
            return "❌ No data loaded"
        
        overview = f"""
📊 **Data Overview**
- **Rows**: {len(self.df):,}
- **Columns**: {len(self.df.columns)}
- **Date Range**: {self.df['InvoiceDate'].min()} to {self.df['InvoiceDate'].max()}
- **Sample Columns**: {', '.join(self.df.columns[:5])}{'...' if len(self.df.columns) > 5 else ''}

💡 **You can ask questions like:**
- "What was the total sales last month?"
- "Show me the top 5 products by revenue"
- "What's the average order value?"
- "Sales trend by month in 2024"
"""
        return overview

    def show_conversation_history(self) -> str:
        """Show recent conversation history"""
        if not self.conversation_history:
            return "💭 No conversation history yet."
        
        history_text = "💭 **Recent Questions:**\n\n"
        for i, entry in enumerate(self.conversation_history[-5:], 1):
            history_text += f"{i}. {entry['user_question']}\n   → {entry['result_summary']}\n\n"
        
        return history_text

# Main Chat Interface
def main():
    """Main chat interface"""
    print("🚀 Initializing CSV Chat Assistant...")
    
    # Initialize the assistant
    api_key = "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"
    csv_file = "sampledata.csv"
    
    assistant = CSVChatAssistant(api_key, csv_file)
    
    if assistant.df is None:
        print("❌ Failed to load data. Exiting...")
        return
    
    print("\n" + "="*60)
    print("🎯 CSV CHAT ASSISTANT")
    print("="*60)
    print(assistant.get_data_overview())
    print("\n💬 **Commands:**")
    print("- Type your question about the data")
    print("- 'overview' - Show data overview")
    print("- 'history' - Show conversation history")
    print("- 'quit' - Exit the assistant")
    print("="*60)
    
    while True:
        try:
            user_input = input("\n🤔 Your question: ").strip()
            
            if not user_input:
                continue
                
            if user_input.lower() in ['quit', 'exit', 'bye']:
                print("👋 Goodbye! Thanks for using CSV Chat Assistant!")
                break
            elif user_input.lower() == 'overview':
                print(assistant.get_data_overview())
                continue
            elif user_input.lower() == 'history':
                print(assistant.show_conversation_history())
                continue
            
            # Process the question
            print("\n🔄 Processing...")
            result = assistant.process_question(user_input)
            print(f"\n📋 **Answer:**\n{result}")
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Unexpected error: {e}")

# Alternative: Class-based approach for integration into other applications
class DataAnalysisAPI:
    """API wrapper for easy integration into web apps or other systems"""
    
    def __init__(self, api_key: str, csv_file_path: str):
        self.assistant = CSVChatAssistant(api_key, csv_file_path)
    
    def ask_question(self, question: str) -> Dict[str, Any]:
        """
        Ask a question and get structured response
        
        Returns:
            {
                'question': str,
                'answer': str,
                'success': bool,
                'timestamp': str,
                'conversation_id': int
            }
        """
        result = self.assistant.process_question(question)
        
        return {
            'question': question,
            'answer': result,
            'success': not result.startswith('❌'),
            'timestamp': datetime.now().isoformat(),
            'conversation_id': len(self.assistant.conversation_history)
        }
    
    def get_conversation_history(self) -> List[Dict[str, Any]]:
        """Get formatted conversation history"""
        return self.assistant.conversation_history
    
    def get_data_info(self) -> Dict[str, Any]:
        """Get information about the loaded data"""
        if self.assistant.df is None:
            return {'error': 'No data loaded'}
        
        return {
            'rows': len(self.assistant.df),
            'columns': len(self.assistant.df.columns),
            'column_names': list(self.assistant.df.columns),
            'date_range': {
                'start': str(self.assistant.df['InvoiceDate'].min()) if 'InvoiceDate' in self.assistant.df.columns else None,
                'end': str(self.assistant.df['InvoiceDate'].max()) if 'InvoiceDate' in self.assistant.df.columns else None
            }
        }

# Example usage and testing
if __name__ == "__main__":
    # For command line interface
    main()
    
    # Example of API usage:
    """
    # For integration into web apps:
    api = DataAnalysisAPI(api_key="your_api_key", csv_file_path="sampledata.csv")
    
    # Ask questions
    response1 = api.ask_question("What was the total sales last month?")
    response2 = api.ask_question("Show me top 5 products by revenue")
    
    print(response1['answer'])
    print(response2['answer'])
    
    # Get conversation history
    history = api.get_conversation_history()
    """

# Enhanced pandas code execution with better error handling
class PandasExecutor:
    """Enhanced pandas code executor with safety and formatting"""
    
    def __init__(self, dataframe: pd.DataFrame):
        self.df = dataframe
    
    def safe_execute(self, code: str) -> Dict[str, Any]:
        """Execute pandas code with enhanced safety and formatting"""
        try:
            # Sanitize code
            code = self.sanitize_code(code)
            
            # Set up safe execution environment
            safe_globals = {
                'df': self.df,
                'pd': pd,
                'datetime': datetime,
                '__builtins__': {
                    'print': print, 'len': len, 'str': str, 'int': int, 
                    'float': float, 'round': round, 'sum': sum, 'min': min, 'max': max
                }
            }
            
            # Execute and capture result
            import io
            import sys
            
            old_stdout = sys.stdout
            sys.stdout = captured_output = io.StringIO()
            
            try:
                # Try to execute as statement first
                exec(code, safe_globals)
                output = captured_output.getvalue().strip()
                
                # If no output, try as expression
                if not output:
                    result = eval(code, safe_globals)
                    return self.format_pandas_result(result)
                else:
                    return {'success': True, 'formatted_result': output, 'type': 'printed'}
                    
            finally:
                sys.stdout = old_stdout
                
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'formatted_result': f"❌ Error: {str(e)}"
            }
    
    def sanitize_code(self, code: str) -> str:
        """Clean and sanitize the pandas code"""
        # Remove markdown code blocks
        code = re.sub(r'^```[^\n]*\n', '', code.strip())
        code = re.sub(r'\n```$', '', code)
        
        # Remove comments and extra whitespace
        lines = [line.strip() for line in code.split('\n') if line.strip() and not line.strip().startswith('#')]
        return '\n'.join(lines)
    
    def format_pandas_result(self, result) -> Dict[str, Any]:
        """Format pandas results with proper styling"""
        if isinstance(result, pd.DataFrame):
            if len(result) == 0:
                formatted = "📊 No data found matching your criteria."
            elif len(result) <= 20:
                formatted = f"📊 **Results** ({len(result)} rows):\n\n{self.format_dataframe(result)}"
            else:
                formatted = f"📊 **Results** (showing first 20 of {len(result)} rows):\n\n{self.format_dataframe(result.head(20))}"
                
        elif isinstance(result, pd.Series):
            if len(result) <= 10:
                formatted = f"📊 **Results**:\n\n{self.format_series(result)}"
            else:
                formatted = f"📊 **Results** (showing first 10 of {len(result)} items):\n\n{self.format_series(result.head(10))}"
                
        elif isinstance(result, (int, float)):
            if isinstance(result, float):
                formatted = f"🔢 **Result**: {result:,.2f}"
            else:
                formatted = f"🔢 **Result**: {result:,}"
                
        else:
            formatted = f"✅ **Result**: {str(result)}"
        
        return {
            'success': True,
            'formatted_result': formatted,
            'type': 'formatted',
            'raw_result': result
        }
    
    def format_dataframe(self, df: pd.DataFrame) -> str:
        """Format DataFrame with nice alignment"""
        # Convert to string with proper formatting
        formatted = df.to_string(index=True, float_format=lambda x: f"{x:,.2f}" if pd.notnull(x) else "")
        return formatted
    
    def format_series(self, series: pd.Series) -> str:
        """Format Series with nice alignment"""
        formatted = series.to_string(float_format=lambda x: f"{x:,.2f}" if pd.notnull(x) else "")
        return formatted

# Example integration script
def create_enhanced_assistant():
    """Create an enhanced version of your existing assistant"""
    
    # Your existing configuration
    api_key = "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"
    csv_file = "sampledata.csv"
    
    # Create the enhanced assistant
    assistant = CSVChatAssistant(api_key, csv_file)
    
    return assistant

# Test the assistant
def test_assistant():
    """Test the assistant with sample questions"""
    assistant = create_enhanced_assistant()
    
    test_questions = [
        "What's the total sales amount?",
        "Show me the top 5 branches by revenue",
        "What's the average order value?",
        "Sales trend by month in 2024",
        "Which product category generates the most revenue?"
    ]
    
    print("🧪 Testing Assistant...")
    for question in test_questions:
        print(f"\n❓ Question: {question}")
        result = assistant.process_question(question)
        print(f"📋 Answer: {result}")

if __name__ == "__main__":
    # Run the main chat interface
    main()