from langgraph.graph import StateGraph,START,END
from src.langgraphagenticai.state.state import State
from src.langgraphagenticai.Nodes.basic_chatbot_node import BasicChatbotNode

class Graphbuilder:
    def __init__(self,model):
        self.llm=model
        self.graph_builder=StateGraph(State)
    def basic_chatbot_build_graph(self):
        """
        Build the basic chatbot graph using langgraph.
        This chatbot initializes the node with 'BasicChatbotNode' class
        and integrates in into the graph. The chatbot node is set as both the entry and exit point of the graph
        """
        self.basic_chatbot_node=BasicChatbotNode(self.llm)
        self.graph_builder.add_node('chatbot',self.basic_chatbot_node.process)
        self.graph_builder.set_entry_point("chatbot")
        self.graph_builder.add_edge(START,'chatbot')
        self.graph_builder.add_edge('chatbot',END)
        
    
    def setup_graph(self,usecase:str):
        """
        setup the graph for the selected use case
        """
        if usecase=="Basic Chatbot":
            self.basic_chatbot_build_graph()
        return self.graph_builder.compile()