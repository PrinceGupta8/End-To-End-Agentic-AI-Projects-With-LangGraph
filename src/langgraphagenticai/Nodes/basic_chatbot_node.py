from src.langgraphagenticai.state.state import State
class BasicChatbotNode:
    """
    Basic chatbot node implementation
    """
    def __init__(self,model):
        self.llm=model

    def process(self,state:State)->dict:
        """
        process the input and generate a chatbot response
        """
        return {'messages':self.llm.invoke(state['messages'])}