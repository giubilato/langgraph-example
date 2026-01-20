from langchain_exa import ExaSearchResults

# Configura Exa. 
tools = [ExaSearchResults(num_results=5, use_autoprompt=True)]
