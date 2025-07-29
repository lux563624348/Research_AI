from langgraph.pregel.remote import RemoteGraph

# Point this URL to your deployed langgraph-api endpoint
graph = RemoteGraph("http://localhost:8123")

# Optionally list tools and metadata
print("Graph metadata:", graph.metadata)

# Run a graph execution
response = graph.invoke({"input": "Hello!"})
print("Response:", response)

