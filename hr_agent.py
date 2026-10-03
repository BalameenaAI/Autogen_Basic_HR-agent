import asyncio
import os
from dotenv import load_dotenv
import autogen

# Load environment variables from the .env file
load_dotenv()

async def main():
    # Configure the LLM settings using the OpenAI API key from environment variables
    llm_config = {
        "config_list": [{
            "model": "gpt-4o-mini",  # You can also use "gpt-4o" 
            "api_key": os.environ.get("OPENAI_API_KEY")
        }],
        "temperature": 0.4
    }

    # 1. Create the HR Assistant Agent with the requested system message
    hr_assistant = autogen.ConversableAgent(
        name="HRAssistant",
        system_message="You are an HR assistant.",
        llm_config=llm_config,
        human_input_mode="NEVER"
    )

    # 2. Create a User Proxy Agent to initiate the dialogue
    user_proxy = autogen.ConversableAgent(
        name="UserProxy",
        human_input_mode="NEVER",
        code_execution_config=False
    )

    # 3. Craft the task text, explicitly resolving the ambiguity of "5 lines" 
    # by demanding 5 precise bullet points.
    task_prompt = (
        "Write a professional job description for a Junior Python Developer. "
        "Constraint: Provide the output as exactly 5 distinct bullet points, "
        "with each bullet point representing a single line."
    )

    print("--- Starting Async Chat with HR Assistant ---")

    # 4. Use asynchronous chat initiation (a_initiate_chat) for the asyncio loop
    chat_result = await user_proxy.a_initiate_chat(
        hr_assistant,
        message=task_prompt,
        max_turns=1
    )

    # 5. Extract and print the final reply from the conversation history
    print("\n--- Final Reply ---")
    if chat_result.chat_history:
        print(chat_result.chat_history[-1]["content"])

if __name__ == "__main__":
    # Run the async main function using asyncio
    asyncio.run(main())