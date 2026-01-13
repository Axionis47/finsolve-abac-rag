"""
Route modules for the RAG chatbot API.
"""
from app.routes.search import router as search_router
from app.routes.chat import router as chat_router
from app.routes.admin import router as admin_router
from app.routes.hr import router as hr_router
from app.routes.auth import router as auth_router

__all__ = ["search_router", "chat_router", "admin_router", "hr_router", "auth_router"]

