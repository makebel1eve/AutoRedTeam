import asyncio
import logging
import os
from dotenv import load_dotenv

load_dotenv()

from openai import AsyncOpenAI
import uvicorn
from pyrit.setup import initialize_pyrit_async
from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import (
    OpenAIChatCompletion,
    OpenAIPromptExecutionSettings,
)
from semantic_kernel.contents import ChatHistory
from semantic_kernel.filters import FilterTypes, PromptRenderContext
from semantic_kernel.prompt_template import PromptTemplateConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
from api import app
from modules.db import ThreatDatabase
from modules.embeddings import Embeddings
from modules.orchestrator import Orchestrator
from modules.semantic_firewall import FirewallPlugin, RiskVerdict


async def main():
    await initialize_pyrit_async(memory_db_type="SQLite")

    embedder = Embeddings()
    db = ThreatDatabase(embedder=embedder)
    await db.init_pool()

    orchestrator = Orchestrator(db=db, embedder=embedder, max_concurrency=5)

    kernel = Kernel()
    firewall = FirewallPlugin(
        db=db,
        embedder=embedder,
        slack=0.2,
        cusum_threshold=1.5,
        w1=0.5,
        w2=0.3,
        w3=0.2,
    )
    chat_completion = OpenAIChatCompletion(
        ai_model_id=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
        api_key=os.getenv("GROQ_API_KEY"),
        async_client=AsyncOpenAI(
            base_url="https://api.groq.com/openai/v1",
            api_key=os.getenv("GROQ_API_KEY"),
        ),
    )
    execution_settings = OpenAIPromptExecutionSettings()
    kernel.add_service(chat_completion)

    kernel.add_function(
        plugin_name="ChatBot",
        function_name="Chat",
        prompt_template_config=PromptTemplateConfig(
            template="{{$chat_history}}{{$user_input}}",
            allow_dangerously_set_content=True,
        ),
        prompt_execution_settings=execution_settings,
    )
    chat_function = kernel.get_function(plugin_name="ChatBot", function_name="Chat")

    @kernel.filter(FilterTypes.PROMPT_RENDERING)
    async def firewall_filter(context: PromptRenderContext, next):
        try:
            user_message = context.arguments.get("user_input", "")
            chat_history: ChatHistory = context.arguments.get(
                "chat_history", ChatHistory()
            )

            last_response = ""
            if chat_history.messages and chat_history.messages[-1].role == "assistant":
                last_response = chat_history.messages[-1].content

            conv_id = context.arguments.get("conversation_id", "demo-session")
            verdict = await firewall.analyze_risk(
                user_message=user_message,
                conversation_id=conv_id,
                last_response=last_response,
            )
            if verdict == RiskVerdict.BLOCK:
                raise Exception("Request blocked by firewall.")

            await next(context)
        except Exception as e:
            print(f"FIREWALL FILTER ERROR: {type(e).__name__}: {e}")
            raise

    app.state.orchestrator = orchestrator
    app.state.kernel = kernel
    app.state.chat_function = chat_function
    app.state.history = ChatHistory()
    app.state.firewall = firewall

    config = uvicorn.Config(app, host="0.0.0.0", port=8000, loop="asyncio")
    server = uvicorn.Server(config)

    logger.info("Dashboard running at http://localhost:8000")

    try:
        await server.serve()
    finally:
        await db.close()


if __name__ == "__main__":
    asyncio.run(main())
