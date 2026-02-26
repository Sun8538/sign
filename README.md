# ASL Sign Language Translation System

A comprehensive real-time American Sign Language (ASL) translation system that enables bidirectional communication between ASL and English. This project uses advanced machine learning, computer vision, and natural language processing to bridge communication gaps.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Next.js](https://img.shields.io/badge/Next.js-14.2-black)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 🎯 Project Overview

This system provides a complete solution for ASL communication with three main features:

1. **Receptive Mode** - Real-time ASL fingerspelling recognition using webcam
2. **Expressive Mode** - Text-to-ASL conversion with 3D avatar animation
3. **Translation Mode** - Bidirectional communication combining both modes

The system uses a combination of MediaPipe for hand landmark detection, TensorFlow/Keras for classification, and LLM integration for ASL Gloss grammar conversion.

---

## ✨ Key Features

### 🎥 Real-Time ASL Fingerspelling Recognition

- **Hand Landmark Detection**: Uses MediaPipe to detect and track 21 hand landmarks in real-time
- **Letter Classification**: Custom-trained neural network (PointNet architecture) classifies fingerspelled letters (A-Z, excluding J and Z which require motion)
- **Smart Recognition Pipeline**:
  - Confidence threshold filtering (default 80%)
  - Letter repetition prevention
  - Automatic space insertion when hand leaves frame
  - Misrecognition correction (e.g., distinguishing between 'A' and 'T')
- **Live Transcription**: Displays recognized text in real-time
- **Spell Checking**: Optional BERT-based spell correction using NeuSpell

### 🤖 Text-to-ASL Avatar Animation

- **3D Avatar Rendering**: Real-time 3D avatar using Three.js and React Three Fiber
- **ASL Gloss Conversion**: LLM-powered conversion of English to ASL grammar structure
  - Converts to Object-Subject-Verb word order
  - Removes unnecessary words (IS, ARE)
  - Replaces pronouns (I → ME)
- **Semantic Sign Matching**: 
  - Vector database (PostgreSQL with pgvector or SQLite fallback)
  - Sentence-transformers for semantic similarity matching
  - Cosine similarity threshold (70%+) for sign selection
- **Fingerspelling Fallback**: Automatically fingerspells words not found in sign database
- **Adjustable Signing Speed**: Control animation playback speed (20-100 FPS)
- **Pause Handling**: Natural pauses for punctuation

### 📊 Advanced ML Architecture

#### Hand Landmark Classification Model
- **Model Type**: PointNet-inspired architecture for 3D point cloud classification
- **Input**: 21 hand landmarks (x, y, z coordinates) from MediaPipe
- **Output**: 24 letter classifications (A-Y, excluding J and Z)
- **Multiple Trained Models Available**:
  - `model5.keras` - Primary production model
  - Various configurations in `/docs/Pointnet Classification/`:
    - Different batch sizes (32, 64, 128)
    - Different epochs (20, 100)
    - Different learning rates (0.0001, 0.0005, 0.001)
    - Best performing: 64 batch, 100 epochs, 0.0005 LR

#### Image Classification Models (Legacy)
- Two CNN-based models trained on image data
- Models with 20 and 30 epochs configurations
- Stored in `/docs/Image Classification Model 1/` and `/docs/Image Classification Model 2/`

### 🗄️ Sign Language Database

- **Database Systems**: Supports both PostgreSQL (with pgvector extension) and SQLite
- **Storage**: Full ASL sign animations stored as landmark coordinate sequences
- **Semantic Search**: Each word has an embedding vector for semantic similarity matching
- **Data Structure**:
  ```sql
  signs (
    word TEXT,           -- English word
    points TEXT,         -- JSON array of landmark coordinates per frame
    embedding VECTOR     -- Semantic embedding for similarity search
  )
  ```

### 🔄 Real-Time Communication

- **WebSocket Protocol**: Flask-SocketIO for real-time bidirectional communication
- **Event-Driven Architecture**:
  - `R-TRANSCRIPTION`: Recognition updates to client
  - `E-ANIMATION`: Animation data to client
  - `E-REQUEST-ANIMATION`: Client requests sign animation
  - `R-CLEAR-TRANSCRIPTION`: Clear recognition history
- **Video Streaming**: MJPEG stream for camera feed preview

### 🧠 LLM Integration

- **Supported Providers**:
  - Azure OpenAI (primary)
  - OpenAI (fallback)
- **ASL Gloss Conversion**: Uses GPT-4o to convert English to proper ASL grammar
- **Graceful Degradation**: Falls back to simple word splitting if LLM unavailable

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         Frontend (Next.js)                       │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │ Recognition  │  │   Express    │  │  Translate   │          │
│  │     Page     │  │     Page     │  │     Page     │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
│         │                  │                  │                  │
│         └──────────────────┴──────────────────┘                 │
│                            │                                     │
│                    ┌───────▼────────┐                           │
│                    │  Socket.IO     │                           │
│                    │    Client      │                           │
│                    └───────┬────────┘                           │
└────────────────────────────┼──────────────────────────────────┘
                             │
                    ┌────────▼─────────┐
                    │   WebSocket      │
                    └────────┬─────────┘
                             │
┌────────────────────────────▼──────────────────────────────────┐
│                     Backend (Flask)                            │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │              Flask-SocketIO Server                       │  │
│  └────────┬────────────────────────────┬───────────────────┘  │
│           │                            │                       │
│  ┌────────▼────────┐        ┌─────────▼────────┐             │
│  │  Recognition    │        │   Expression     │             │
│  │    Pipeline     │        │    Pipeline      │             │
│  └────────┬────────┘        └─────────┬────────┘             │
│           │                            │                       │
│  ┌────────▼────────┐        ┌─────────▼────────┐             │
│  │  Landmarker     │        │      LLM         │             │
│  │  (MediaPipe)    │        │   (ASL Gloss)    │             │
│  └────────┬────────┘        └─────────┬────────┘             │
│           │                            │                       │
│  ┌────────▼────────┐        ┌─────────▼────────┐             │
│  │  Classifier     │        │  Sign Database   │             │
│  │ (TensorFlow)    │        │  (PostgreSQL/    │             │
│  │                 │        │    SQLite)       │             │
│  └────────┬────────┘        └─────────┬────────┘             │
│           │                            │                       │
│  ┌────────▼────────┐        ┌─────────▼────────┐             │
│  │   BERT Spell    │        │  Semantic        │             │
│  │   Checker       │        │  Matching        │             │
│  │  (Optional)     │        │ (Transformers)   │             │
│  └─────────────────┘        └──────────────────┘             │
└─────────────────────────────────────────────────────────────┘
```

---

## 🛠️ Technology Stack

### Frontend
- **Framework**: Next.js 14.2 (React 18)
- **UI Library**: Tailwind CSS, Radix UI components
- **3D Graphics**: Three.js, React Three Fiber
- **WebSocket**: Socket.IO Client
- **Speech Recognition**: react-speech-recognition (Web Speech API)
- **MediaPipe**: Hand landmark detection in browser
- **Avatar Animation**: Custom 3D avatar with pose and hand rigging

### Backend
- **Framework**: Flask with Flask-SocketIO
- **ML Framework**: TensorFlow 2.x, Keras
- **Computer Vision**: OpenCV, MediaPipe
- **NLP**: 
  - LangChain (LLM integration)
  - Sentence Transformers (semantic embeddings)
  - NeuSpell (spell checking)
- **Database**: 
  - PostgreSQL with pgvector extension
  - SQLite (fallback)
- **LLM Integration**:
  - Azure OpenAI
  - OpenAI API

### Development Tools
- **Language**: Python 3.10+, TypeScript
- **Package Managers**: pip, npm
- **Environment**: Virtual environments (.venv)

---

## 📁 Project Structure

```
ASL_sign/
├── .venv/                              # Python virtual environment
├── requirements.txt                    # Python dependencies
├── SETUP_GUIDE.md                     # Detailed setup instructions
├── README.md                          # This file
│
├── docs/                              # Trained models and documentation
│   ├── Image Classification Model 1/  # CNN-based classification model
│   ├── Image Classification Model 2/  # Alternative CNN model
│   └── Pointnet Classification/       # PointNet models (various configs)
│       ├── Architecture 2 - Batch 128 - 100 Epochs - 0.0005 LR/
│       ├── Batch 32 - 20 Epochs - 0.001 LR/
│       ├── Batch 64 - 100 Epochs - 0.0005 LR/
│       └── ...
│
├── paper/                             # Research paper (LaTeX)
│   ├── main.tex
│   ├── refs.bib
│   └── documents/
│
└── src/
    ├── client/                        # Next.js frontend application
    │   ├── package.json
    │   ├── next.config.mjs
    │   ├── tailwind.config.ts
    │   ├── tsconfig.json
    │   │
    │   ├── public/
    │   │   └── landmarker/            # MediaPipe models
    │   │
    │   └── src/
    │       ├── app/
    │       │   ├── page.tsx           # Main translation page
    │       │   ├── express/
    │       │   │   └── page.tsx       # Express mode (text → ASL)
    │       │   ├── translate/
    │       │   │   └── page.tsx       # Bidirectional translation
    │       │   │
    │       │   └── components/
    │       │       ├── Avatar.tsx     # 3D avatar rendering
    │       │       ├── Camera.tsx     # Webcam component
    │       │       ├── Transcription.tsx  # Text display
    │       │       ├── Visualization.tsx  # Avatar container
    │       │       └── lib.ts         # 3D drawing utilities
    │       │
    │       ├── lib/                   # Utility functions
    │       └── ui/                    # UI components
    │
    └── server/                        # Flask backend server
        ├── server.py                  # Main server application
        ├── .env                       # Environment configuration (create this)
        │
        ├── model5.keras               # Primary classification model
        ├── model.h5, model2-6.keras   # Alternative models
        │
        ├── alphabets/                 # Fingerspelling animation data
        │   ├── A.json, B.json, ...
        │   └── Z.json                 # 26 letter animations
        │
        ├── scripts/                   # Data processing scripts
        │   ├── convert_dataset_to_nets.py     # Training data prep
        │   ├── convert_videos_to_points.py    # Extract landmarks from videos
        │   ├── create_dataset_directories.py  # Dataset organization
        │   ├── embed_database_words.py        # Generate word embeddings
        │   ├── film_alphabet_frames.py        # Capture fingerspelling
        │   ├── scrape_videos.py               # Download sign videos
        │   └── raw1.py, raw2.py, raw3.py      # Data preprocessing
        │
        └── utils/                     # Core utilities
            ├── __init__.py
            ├── bert.py                # BERT spell checker
            ├── classifier.py          # ML model inference
            ├── landmarker.py          # MediaPipe landmark detection
            ├── llm.py                 # LLM integration (ASL Gloss)
            ├── recognition.py         # Recognition pipeline
            └── store.py               # State management
```

---

## 🚀 Installation & Setup

### Prerequisites

- **Python**: 3.10, 3.11, or 3.12
- **Node.js**: 18.x or higher
- **npm**: 9.x or higher
- **Webcam**: Required for ASL recognition
- **PostgreSQL** (Optional): For production database with vector search
- **LLM API Key** (Optional): Azure OpenAI or OpenAI for ASL Gloss conversion

### Quick Start

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd ASL_sign
   ```

2. **Set up Python environment**
   ```bash
   # Create virtual environment
   python -m venv .venv
   
   # Activate (Windows PowerShell)
   .\.venv\Scripts\Activate.ps1
   
   # Activate (Mac/Linux)
   source .venv/bin/activate
   
   # Install dependencies
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

3. **Set up Node.js environment**
   ```bash
   cd src/client
   npm install
   cd ../..
   ```

4. **Configure environment variables**
   
   Create `src/server/.env`:
   ```env
   # Azure OpenAI (Recommended)
   AZURE_OPENAI_API_KEY=your_api_key
   AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
   AZURE_OPENAI_DEPLOYMENT_NAME=gpt-4o
   AZURE_OPENAI_API_VERSION=2024-08-01-preview
   
   # OR OpenAI (Alternative)
   OPENAI_API_KEY=your_openai_key
   
   # PostgreSQL (Optional - falls back to SQLite)
   POSTGRES_HOST=localhost
   POSTGRES_PORT=5432
   POSTGRES_DB=signs
   POSTGRES_USER=postgres
   POSTGRES_PASSWORD=your_password
   ```

5. **Start the backend server**
   ```bash
   # From project root with venv activated
   cd src/server
   python server.py
   ```
   
   Server will start on `http://localhost:1234`

6. **Start the frontend client** (in a new terminal)
   ```bash
   cd src/client
   npm run dev
   ```
   
   Client will start on `http://localhost:3000`

7. **Access the application**
   - Main Page: http://localhost:3000
   - Express Mode: http://localhost:3000/express
   - Translate Mode: http://localhost:3000/translate

---

## 📖 Usage Guide

### Recognition Mode (ASL → English)

1. Navigate to http://localhost:3000
2. Allow camera access when prompted
3. Sign fingerspelled letters in front of the camera
4. View real-time transcription below the video feed
5. Options:
   - **Autocorrect**: Enable BERT-based spell checking
   - **Clear**: Reset transcription

**Tips for Best Results**:
- Use good lighting
- Keep hand within camera frame
- Use solid background
- Hold each letter for 1-2 seconds
- Remove hand from frame to insert spaces

### Express Mode (English → ASL)

1. Navigate to http://localhost:3000/express
2. Enter text in the "Content" textarea
3. Adjust signing speed (20-100 FPS)
4. Optional: Set duration in seconds for auto-speed
5. Click "Render" to generate animation
6. Watch the 3D avatar perform the signs

**System Behavior**:
- Converts English to ASL Gloss grammar automatically
- Uses semantic matching to find appropriate signs
- Falls back to fingerspelling for unknown words
- Adds natural pauses for punctuation

### Translation Mode (Bidirectional)

1. Navigate to http://localhost:3000/translate
2. Use left panel for ASL recognition
3. Use right panel for text-to-ASL expression
4. Both modes operate simultaneously
5. Enable voice input for hands-free English input

---

## 🧪 Development & Training

### Data Collection Scripts

The project includes several scripts for collecting and processing ASL data:

#### 1. Video Scraping
```bash
cd src/server/scripts
python scrape_videos.py
```
Downloads ASL sign videos from online sources for database population.

#### 2. Video to Landmarks Conversion
```bash
python convert_videos_to_points.py
```
Processes videos to extract MediaPipe landmarks:
- Extracts 33 pose landmarks
- Extracts 21 landmarks per hand (left/right)
- Stores frame-by-frame coordinate data
- Saves to database for animation playback

#### 3. Database Embedding Generation
```bash
python embed_database_words.py
```
Generates semantic embeddings for all words in database:
- Uses SentenceTransformer (all-MiniLM-L6-v2)
- Creates 384-dimensional vectors
- Enables semantic similarity search

#### 4. Fingerspelling Capture
```bash
python film_alphabet_frames.py
```
Records fingerspelling animations for A-Z letters.

#### 5. Dataset Preparation
```bash
python create_dataset_directories.py  # Organize training data
python convert_dataset_to_nets.py     # Convert to model input format
```

### Model Training

The project includes multiple model architectures trained with various hyperparameters:

**Best Performing Model**: 
- Architecture: PointNet-inspired
- Batch Size: 64
- Epochs: 100
- Learning Rate: 0.0005
- Location: `docs/Pointnet Classification/Batch 64 - 100 Epochs - 0.0005 LR/`

**Training Variations**:
- Batch sizes: 32, 64, 128
- Epochs: 20, 100
- Learning rates: 0.0001, 0.0005, 0.001

All models accept normalized 3D landmark coordinates (21 points × 3 dimensions) and output 24-class predictions.

---

## 🔧 Configuration Options

### Recognition Pipeline

```python
# In utils/recognition.py
Recognition(min_confidence=0.80)  # Adjust confidence threshold
```

### Landmark Detection

```python
# In utils/landmarker.py
Landmarker(
    model_complexity=0,              # 0=lite, 1=full (default: 0)
    min_detection_confidence=0.75,   # Detection threshold
    min_tracking_confidence=0.75,    # Tracking threshold
    max_num_hands=1                  # Maximum hands to detect
)
```

### Classification Model

Swap models by changing in `server.py`:
```python
# Load different model
model = keras.models.load_model("model6.keras")  # Or model2.keras, etc.
```

### Database Configuration

The system automatically handles:
- PostgreSQL with pgvector (preferred for semantic search)
- SQLite fallback (automatic if PostgreSQL unavailable)
- No code changes needed for fallback

---

## 🎨 Frontend Components

### Main Components

1. **Camera**: Webcam feed display with MediaPipe overlay
2. **Avatar**: 3D character rendering with Three.js
3. **Visualization**: Animation container with signing speed control
4. **Transcription**: Text display for recognized/translated content

### Styling

- Built with Tailwind CSS
- Responsive design
- Dark theme optimized for readability
- Custom Radix UI components

---

## 🔍 Core Algorithms

### Recognition Algorithm

```
1. Capture frame from webcam
2. Convert to RGB, run MediaPipe hand detection
3. Extract 21 hand landmarks (x, y, z)
4. Normalize coordinates to 0-1 range
5. Pass to neural network classifier
6. Get letter prediction with confidence score
7. Apply confidence threshold (default 80%)
8. Implement letter repetition logic:
   - Check last 20 frames for consistency
   - Prevent double letters unless intentional
   - Check last 4 frames for new letter
9. Apply misrecognition corrections (A vs T)
10. Add to transcription buffer
11. When hand leaves frame:
    - Mark end of word
    - Run BERT spell checker (optional)
    - Update corrected transcription
12. Emit to frontend via WebSocket
```

### Expression Algorithm

```
1. Receive text from client
2. Convert to ASL Gloss using LLM:
   - Reorder to Object-Subject-Verb
   - Remove "is", "are"
   - Replace "I" with "ME"
3. Split into words
4. For each word:
   a. Generate semantic embedding (384-dim vector)
   b. Query database with cosine similarity
   c. If similarity > 70%:
      - Use stored sign animation
   d. Else:
      - Use fingerspelling animation
5. Add pauses for punctuation
6. Emit animation sequence to client
7. Client renders with 3D avatar
```

### Semantic Matching

```sql
-- PostgreSQL query
SELECT word, points, (embedding <=> query_embedding) AS distance
FROM signs
ORDER BY distance ASC
LIMIT 1;

-- cosine_similarity = 1 - distance
-- threshold: similarity > 0.70
```

---

## 🗄️ Database Schema

### PostgreSQL Setup

```sql
-- Create database
CREATE DATABASE signs;

-- Enable pgvector extension
CREATE EXTENSION vector;

-- Create table
CREATE TABLE signs (
    id SERIAL PRIMARY KEY,
    word TEXT NOT NULL,
    points TEXT NOT NULL,           -- JSON array of landmark sequences
    embedding VECTOR(384)            -- Semantic embedding
);

-- Create index for fast similarity search
CREATE INDEX ON signs USING ivfflat (embedding vector_cosine_ops);
```

### SQLite Fallback

```sql
CREATE TABLE IF NOT EXISTS signs (
    word TEXT,
    points TEXT,
    embedding BLOB                   -- Serialized numpy array
);
```

---

## 🧪 Testing & Validation

### Manual Testing Checklist

**Recognition Mode**:
- [ ] Camera feed displays correctly
- [ ] Hand landmarks render in real-time
- [ ] Letters are recognized with >80% confidence
- [ ] Spaces are inserted when hand leaves frame
- [ ] Spell checking corrects common errors
- [ ] Clear button resets transcription

**Express Mode**:
- [ ] Text input accepts all characters
- [ ] Animation renders for known words
- [ ] Fingerspelling works for unknown words
- [ ] Signing speed adjustment works
- [ ] Avatar moves smoothly
- [ ] Pauses for punctuation

**Translation Mode**:
- [ ] Both panels work simultaneously
- [ ] Voice input captures speech
- [ ] Camera feed and avatar both render

### Performance Benchmarks

- **Recognition Latency**: ~50-100ms per frame
- **Classification Time**: ~10-20ms
- **LLM Gloss Conversion**: ~500-2000ms (network dependent)
- **Database Query**: ~5-10ms (PostgreSQL), ~1-2ms (SQLite)
- **Frame Rate**: 15-30 FPS (dependent on hardware)

---

## 🚧 Known Limitations

1. **Letter Coverage**: J and Z require motion (not supported in still fingerspelling)
2. **Sign Database**: Limited vocabulary (~common words only)
3. **Lighting**: Poor lighting affects landmark detection accuracy
4. **Hand Orientation**: Works best with palm facing camera
5. **Multiple Hands**: System configured for single-hand detection
6. **Background**: Busy backgrounds may interfere with detection
7. **ASL Gloss**: LLM conversion may not be grammatically perfect
8. **Spell Checker**: NeuSpell may fail to load on some systems

---

## 🔮 Future Enhancements

### Planned Features
- [ ] Support for J and Z (motion-based letters)
- [ ] Two-handed sign support
- [ ] Expand sign vocabulary database
- [ ] Real-time grammar correction
- [ ] Mobile app development
- [ ] Multi-user video chat with ASL overlay
- [ ] Sign language learning mode
- [ ] Export transcription to text file
- [ ] Integration with video conferencing platforms

### Research Directions
- [ ] Improve ASL Gloss conversion accuracy
- [ ] Continuous sign language recognition (not just fingerspelling)
- [ ] Context-aware sign selection
- [ ] User-specific model fine-tuning
- [ ] Reduced model size for edge deployment
- [ ] Multi-language sign language support

---

## 🤝 Contributing

Contributions are welcome! Areas for contribution:

1. **Data Collection**: Record more ASL sign videos
2. **Model Training**: Experiment with new architectures
3. **Sign Database**: Add more words to the database
4. **UI/UX**: Improve frontend design and user experience
5. **Documentation**: Improve setup guides and tutorials
6. **Testing**: Report bugs and edge cases

---

## 📄 License

This project is released under the MIT License. See LICENSE file for details.

---

## 🙏 Acknowledgments

- **MediaPipe**: Hand landmark detection
- **TensorFlow**: Deep learning framework
- **LangChain**: LLM integration framework
- **OpenAI/Azure**: GPT models for ASL Gloss
- **pgvector**: Vector similarity search
- **ASL Community**: For sign language resources and data

---

## 📞 Support

For issues and questions:
1. Check [SETUP_GUIDE.md](SETUP_GUIDE.md) for detailed setup instructions
2. Review the Troubleshooting section
3. Check browser console (F12) for client-side errors
4. Check terminal output for server-side errors
5. Ensure all prerequisites are installed correctly

---

## 📚 References

### Technologies
- [MediaPipe Hands](https://developers.google.com/mediapipe/solutions/vision/hand_landmarker)
- [TensorFlow/Keras](https://www.tensorflow.org/)
- [Next.js Documentation](https://nextjs.org/docs)
- [Flask-SocketIO](https://flask-socketio.readthedocs.io/)
- [Three.js](https://threejs.org/)
- [pgvector](https://github.com/pgvector/pgvector)
- [LangChain](https://python.langchain.com/)

### ASL Resources
- [ASL University](https://www.lifeprint.com/)
- [HandSpeak ASL Dictionary](https://www.handspeak.com/)
- [National Association of the Deaf](https://www.nad.org/)

---

## 📊 Project Statistics

- **Total Lines of Code**: ~5,000+
- **Languages**: Python, TypeScript, TSX
- **Dependencies**: 30+ Python packages, 25+ npm packages
- **Model Parameters**: ~1M+ (PointNet architecture)
- **Supported Letters**: 24 (A-Y, excluding J and Z)
- **Database Size**: Depends on sign vocabulary (expandable)

---


