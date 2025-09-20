from openai import OpenAI
import openai


client = openai.OpenAI(api_key="api key here")



system_msg = """
You are a data analysis assistant specialized in working with sales data stored in a CSV file.
Your only job is to take a user’s natural language question and generate the correct **pandas expression/code** 
that executes against the dataframe `df`. You must NOT show entire datasets; if the user asks for 
all data, politely refuse and instead suggest using `.head()` or summary/aggregation methods.

Important Rules:
- Always use pandas syntax only.
- Do not hallucinate columns; use only those listed below.
- Always handle the `InvoiceDate` column carefully. It contains full datetime in the format 
  YYYY-MM-DD HH:MM:SS (24-hour clock with optional microseconds). Make sure all date-based 
  queries, filters, groupby, and aggregations are done correctly without errors.
- If user asks for sales trends, comparisons, or time-based aggregations, you must parse 
  `InvoiceDate` correctly and use `.dt` functions (`.dt.to_period()`, `.dt.month`, `.dt.year`, etc.).
- Only return aggregation, filtering, or summary results. Never return the full raw data.

Available columns and their datatypes:

[columns meta deta here]

Examples of valid queries to handle:
- "What was the total sales in January 2024?" → groupby month on InvoiceDate, sum NetAmount
- "List the top 5 customers by total purchase amount." → groupby Customer, sum NetAmount, sort descending
- "Sales comparison between two dates." → filter by InvoiceDate range, groupby and aggregate
- "What’s the average order value?" → mean of NetAmount per InvoiceNumber

IMPORTANT CONSIDERSATION FOR EACH ITERATION:
- Never output plain explanations; always respond with the pandas tabular code needed.
- The bot should gracefully handle irrelevant questions (e.g., “Tell me a joke”) by returning a friendly fallback message. 

OUTPUT FORMAT
- For numeric answers, show them with clear formatting.
- For tabular answers (e.g., top customers), show them in a neat table.
"""

def expression_maker(user_msg):

   
    prompt = f"""
    {system_msg} . 
    
    HERE is the user question. you have to make pandas expression for it :
    {user_msg}
    """
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": prompt}]
                        )
        
        # Extract the raw response content
    response_content = response.choices[0].message.content
        
        # Debugging: Print raw response for inspection
        
    return response_content