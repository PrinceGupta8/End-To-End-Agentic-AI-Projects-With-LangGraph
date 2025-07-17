import streamlit as st
from src.langgraphagenticai.ui.steamlitui.loadui import LoadStreamlitUI
from src.langgraphagenticai.LLM.groqllm import GroqLLM
from src.langgraphagenticai.Graph.graph_builder import Graphbuilder
from src.langgraphagenticai.ui.steamlitui.display_result import DisplayResultStreamlit

def load_streamlit_agentic_ai_app():
    """
    Loads and runs the agentic ai application with streamlit UI. This function intializes the ui,
    handel user inputs, configures the llm model , sets up the graph based on selected usecase
    displays the output while implementing exception handling for robustness
    """
    ui=LoadStreamlitUI()
    user_input=ui.load_streamlit_ui()
    if not user_input:
        st.error("Error: Falild to load user input from the UI")
        return
    
    if st.session_state.IsFetchButtonClicked:
        user_message=st.session_state.timeframe
    else:
        user_message=st.chat_input("Enter your message")
    if user_message:
        try:
            obj_llm_config=GroqLLM(user_controls_input=user_input)
            model=obj_llm_config.get_llm_model()
            if not model:
                st.error("Error: LLM model could not be initialized")
                return 
            usecase=user_input.get("selected_usecase")
            if not usecase:
                st.error("Error: No usecase selected")
                return 
            
            # Graph builder
            graph_builder=Graphbuilder(model)
            try:
                graph=graph_builder.setup_graph(usecase)
                DisplayResultStreamlit(usecase,graph,user_message).display_result_on_ui()
            except Exception as e:
                st.error(e)
                return
        except Exception as e:
            print(f"Error:{e}")
            return