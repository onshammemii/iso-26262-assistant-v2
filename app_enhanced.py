"""
Flask-based ISO 26262 Safety Assistant
FIXED - Correct filename and compatibility with Hugging Face Spaces
"""

from flask import Flask, render_template, request, jsonify, session
from flask_session import Session
import os
import uuid
from datetime import datetime
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import RAG chain
try:
    from rag_chain_enhanced import ConversationalRAGChain
except ImportError as e:
    print(f"❌ ERROR importing RAG chain: {e}")
    print("Make sure file is named 'rag_chain_enhanced.py'")
    raise

from vector_store import load_or_create_vector_store

app = Flask(__name__)
app.secret_key = os.urandom(24)

# Configure session
app.config["SESSION_PERMANENT"] = False
app.config["SESSION_TYPE"] = "filesystem"
Session(app)

# Global RAG chain instance
rag_chain = None
vector_store_loaded = False


def initialize_rag_system():
    """Initialize the RAG system once"""
    global rag_chain, vector_store_loaded

    if rag_chain is not None:
        return True

    try:
        print("\n" + "="*50)
        print("🚀 Initializing RAG System...")
        print("="*50)

        # Load API key
        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            print("❌ ERROR: GROQ_API_KEY not found in environment")
            print("💡 Set it with: export GROQ_API_KEY='your_key'")
            return False

        print(f"✅ API key found: {groq_api_key[:10]}...{groq_api_key[-10:]}")

        # Load vector store
        print("\n📦 Loading vector store...")
        vector_store = load_or_create_vector_store()

        if vector_store is None:
            print("❌ ERROR: Failed to load vector store")
            return False

        print("✅ Vector store loaded successfully")

        # Initialize RAG chain
        print("\n🤖 Initializing RAG chain...")
        rag_chain = ConversationalRAGChain(
            vector_store=vector_store,
            groq_api_key=groq_api_key,
            model_name="llama-3.1-8b-instant"
        )

        vector_store_loaded = True
        print("\n" + "="*50)
        print("✅ RAG system initialized successfully!")
        print("="*50 + "\n")
        return True

    except Exception as e:
        print(f"\n❌ ERROR initializing RAG system: {e}")
        import traceback
        traceback.print_exc()
        return False


def get_or_create_conversations():
    """Get or create conversations dict in session"""
    if 'conversations' not in session:
        session['conversations'] = {}
    return session['conversations']


def get_or_create_active_conversation():
    """Get or create active conversation ID"""
    if 'active_conversation_id' not in session:
        conv_id = str(uuid.uuid4())
        conversations = get_or_create_conversations()
        conversations[conv_id] = {
            'id': conv_id,
            'title': 'New Chat',
            'messages': [],
            'created_at': datetime.now().isoformat(),
            'updated_at': datetime.now().isoformat()
        }
        session['active_conversation_id'] = conv_id
        session.modified = True

    return session['active_conversation_id']


def get_conversation_history(conversation_id):
    """Get formatted conversation history for RAG"""
    conversations = get_or_create_conversations()
    if conversation_id not in conversations:
        return []

    messages = conversations[conversation_id]['messages']
    history = []
    for msg in messages:
        history.append({
            'role': msg['role'],
            'content': msg['content']
        })

    return history


def generate_conversation_title(first_message):
    """Generate a short title from the first message"""
    title = first_message.replace('\n', ' ')[:50]
    if len(first_message) > 50:
        title += "..."
    return title


@app.route('/')
def home():
    """Home page with example questions"""
    conversations = get_or_create_conversations()
    active_id = session.get('active_conversation_id')

    show_chat = False
    if active_id and active_id in conversations:
        if len(conversations[active_id]['messages']) > 0:
            show_chat = True

    return render_template('index.html', show_chat=show_chat)


@app.route('/api/init', methods=['GET'])
def api_init():
    """Initialize RAG system and return status"""
    success = initialize_rag_system()

    conversations = get_or_create_conversations()
    active_id = get_or_create_active_conversation()

    return jsonify({
        'success': success,
        'system_ready': vector_store_loaded,
        'conversations': list(conversations.values()),
        'active_conversation_id': active_id
    })


@app.route('/api/query', methods=['POST'])
def api_query():
    """Process a query with conversation context"""
    if not vector_store_loaded:
        return jsonify({'error': 'System not initialized'}), 500

    data = request.json
    question = data.get('question', '').strip()
    num_sources = data.get('num_sources', 12)

    if not question:
        return jsonify({'error': 'Question is required'}), 400

    try:
        conv_id = get_or_create_active_conversation()
        conversations = get_or_create_conversations()
        conversation = conversations[conv_id]

        if len(conversation['messages']) == 0:
            conversation['title'] = generate_conversation_title(question)

        history = get_conversation_history(conv_id)

        result = rag_chain.query(
            question=question,
            chat_history=history,
            k=num_sources
        )

        conversation['messages'].append({
            'role': 'user',
            'content': question,
            'timestamp': datetime.now().isoformat()
        })

        conversation['messages'].append({
            'role': 'assistant',
            'content': result['answer'],
            'sources': result['sources'],
            'used_context': result.get('used_context', False),
            'contextualized_question': result.get('contextualized_question'),
            'timestamp': datetime.now().isoformat()
        })

        conversation['updated_at'] = datetime.now().isoformat()
        session.modified = True

        return jsonify({
            'success': True,
            'answer': result['answer'],
            'sources': result['sources'],
            'used_context': result.get('used_context', False),
            'contextualized_question': result.get('contextualized_question'),
            'conversation_id': conv_id,
            'conversation_title': conversation['title']
        })

    except Exception as e:
        print(f"❌ Error processing query: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/conversations', methods=['GET'])
def api_get_conversations():
    """Get all conversations"""
    conversations = get_or_create_conversations()
    active_id = get_or_create_active_conversation()

    return jsonify({
        'conversations': list(conversations.values()),
        'active_conversation_id': active_id
    })


@app.route('/api/conversations/new', methods=['POST'])
def api_new_conversation():
    """Create a new conversation"""
    conv_id = str(uuid.uuid4())
    conversations = get_or_create_conversations()

    conversations[conv_id] = {
        'id': conv_id,
        'title': 'New Chat',
        'messages': [],
        'created_at': datetime.now().isoformat(),
        'updated_at': datetime.now().isoformat()
    }

    session['active_conversation_id'] = conv_id
    session.modified = True

    return jsonify({
        'success': True,
        'conversation_id': conv_id
    })


@app.route('/api/conversations/<conversation_id>/activate', methods=['POST'])
def api_activate_conversation(conversation_id):
    """Switch to a different conversation"""
    conversations = get_or_create_conversations()

    if conversation_id not in conversations:
        return jsonify({'error': 'Conversation not found'}), 404

    session['active_conversation_id'] = conversation_id
    session.modified = True

    return jsonify({
        'success': True,
        'conversation': conversations[conversation_id]
    })


@app.route('/api/conversations/<conversation_id>', methods=['GET'])
def api_get_conversation(conversation_id):
    """Get a specific conversation"""
    conversations = get_or_create_conversations()

    if conversation_id not in conversations:
        return jsonify({'error': 'Conversation not found'}), 404

    return jsonify(conversations[conversation_id])


@app.route('/api/conversations/<conversation_id>', methods=['DELETE'])
def api_delete_conversation(conversation_id):
    """Delete a conversation"""
    conversations = get_or_create_conversations()

    if conversation_id not in conversations:
        return jsonify({'error': 'Conversation not found'}), 404

    del conversations[conversation_id]

    if session.get('active_conversation_id') == conversation_id:
        if len(conversations) > 0:
            session['active_conversation_id'] = list(conversations.keys())[0]
        else:
            new_id = str(uuid.uuid4())
            conversations[new_id] = {
                'id': new_id,
                'title': 'New Chat',
                'messages': [],
                'created_at': datetime.now().isoformat(),
                'updated_at': datetime.now().isoformat()
            }
            session['active_conversation_id'] = new_id

    session.modified = True

    return jsonify({'success': True})


if __name__ == '__main__':
    print("\n" + "🛡️  "*20)
    print("    ISO 26262 Safety Assistant - Starting...")
    print("🛡️  "*20 + "\n")

    # Get port from environment (Hugging Face Spaces uses 7860)
    port = int(os.getenv('PORT', 7860))
    
    if not initialize_rag_system():
        print("\n⚠️  WARNING: RAG system failed to initialize!")
        print("The app will start but won't be able to answer questions.")
        print("Please check the errors above and restart.\n")

    print("\n🌐 Starting Flask server...")
    print(f"👉 Running on: http://0.0.0.0:{port}")
    print("\n💡 Press CTRL+C to stop\n")

    app.run(debug=False, host='0.0.0.0', port=port)
