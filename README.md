# Automated Technical Session Creation Package

This repository contains a comprehensive suite of tools for automatically creating and organizing technical sessions for academic conferences using AI-powered similarity analysis. The system uses machine learning embeddings to group presentations by topic similarity, supports hybrid sessions with pre-assigned invited speakers, and provides interactive tools for session analysis and visualization.

## 🚀 Overview

The Session Creation Package automates the traditionally manual process of organizing conference presentations into thematic sessions. It leverages state-of-the-art sentence transformer models to:

- **Generate semantic embeddings** from presentation titles and abstracts
- **Cluster presentations** into coherent thematic sessions
- **Support hybrid sessions** with pre-assigned invited presentations
- **Analyze session quality** using multiple similarity metrics
- **Generate session titles and keywords** using large language models
- **Match sessions to committees** based on topic similarity
- **Provide interactive visualization** tools for session exploration

## 📁 Repository Structure

### Core Programs

#### 🔧 `session_organizer.py`
The main algorithmic engine containing all core functionality:

- **Data Loading Functions**: Load presentations, hybrid sessions, and committee data from Excel/CSV files
- **Embedding Generation**: Create semantic embeddings using SentenceTransformer models
- **Duplicate Detection**: Remove near-duplicate presentations based on similarity thresholds
- **Session Creation**: Advanced hierarchical clustering with hybrid session support
- **Quality Analysis**: Calculate session coherence, distinctiveness, and presentation-session fit metrics
- **Content Generation**: Generate session titles and keywords using Ollama, local LLaMA, or Gemini models
- **Committee Matching**: Find relevant committees for each session based on topic similarity

#### 🖥️ `session_creation_app_v2.py`
GUI desktop application for interactive session creation:

- **File Management**: Load presentation data, hybrid sessions, and committee information
- **Model Selection**: Choose from multiple embedding models (MiniLM, MPNet, CDE, etc.)
- **Parameter Configuration**: Adjust clustering parameters, similarity thresholds, and session constraints
- **Real-time Processing**: Live output capture and progress monitoring
- **Export Options**: Save results in multiple formats (CSV, Parquet, encrypted)
- **Session Analysis**: Built-in quality metrics and visualization

#### 🌐 `session_creation_viewer_web_app.py`
Streamlit web application for session exploration and analysis:

- **Interactive Visualization**: Explore presentation and session similarities
- **Secure Access**: Password-protected abstract viewing
- **Similarity Metrics**: Detailed explanations of all quality measures
- **Data Filtering**: Search and filter sessions and presentations
- **Export Capabilities**: Download filtered results and analysis reports

#### 📊 `session creation operation.ipynb`
Comprehensive Jupyter notebook demonstrating the complete workflow:

- **Step-by-step Examples**: Complete workflows for different scenarios
- **Hybrid Session Integration**: Adding invited presentations to generated sessions
- **Quality Analysis**: Detailed session evaluation and metrics calculation
- **Comparison Studies**: Analyze algorithm-generated vs. manually-edited sessions
- **Data Export**: Prepare data for web visualization and sharing

### Data Files

#### Input Data Examples
*These raw data files are not provided in the public repository. The notebook data processing example references these files, so this is a description of what is in each file.*
- **`1.29.25 Abstracts.xlsx/csv`**: Sample presentation data with titles, abstracts, and metadata
- **`Example Hybrid Session Invited Presentations.csv`**: Pre-assigned presentations for hybrid sessions
- **`ASABE Committees.csv`**: Committee information for session matching
- **`Presentations - Manual Placement.csv`**: Manually edited session assignments for comparison

#### Output Data Examples
*These example files are not in the public repository but are produced by the processing steps.*
##### Outputs Files for Human Organizers
- **`created_sessions.csv`**: Generated session assignments and metadata
- **`normal_presentations_processed.csv`**: Processed presentation data with session assignments
##### Output Files for the Streamlit Web Application
- **`df_sessions.parquet`**: Session metadata and statistics
- **`pres_similarities_matrix.parquet`**: Presentation-to-presentation similarity matrix
- **`session_similarities_matrix.parquet`**: Session-to-session similarity matrix
- **`encrypted_df.crypt`**: Encrypted full dataset with sensitive information

### Supporting Files
- **`Automated Technical Session Creation Flyer.html`**: Project documentation and flyer
#### Additional Files
- **`llama-3.2-3b-instruct-q8_0.gguf`**: Local LLaMA model for title generation. 
*Necessary for local title generation using SentenceTransformers instead of Ollama. Sentence Transformers can also automatically download if not in the local directory. It is not included, but can be downloaded from Hugging Face.*
- **`.env`**: Environment variables for API keys and passwords. 
*Necessary to create the encryted dataframes for the Streamlit Web Application or to use Gemini to create session titles.*


## 🔧 Installation & Setup

### Prerequisites
```bash
pip install pandas numpy scikit-learn sentence-transformers
pip install streamlit tkinter cryptpandas python-dotenv
pip install google-genai requests pydantic
```

### Environment Configuration
Create a `.env` file with:
```env
GOOGLE_API_KEY=your_gemini_api_key
DATAFRAME_PW=your_encryption_password
```

### Optional: Ollama Setup
For local LLM title generation with Ollama:
1. Install [Ollama](https://ollama.ai/)
2. Run: `ollama pull llama3.2:latest`
3. Start server: `ollama serve`

## 🚀 Usage Examples

### Basic Session Creation (GUI)
```bash
python session_creation_app_v2.py
```
1. Load presentation data (Excel/CSV)
2. Select embedding model
3. Configure session parameters
4. Run session creation
5. Export results

### Command Line Processing (Notebook)
Open `session creation operation.ipynb` and follow the step-by-step workflow for:
- Standard session creation
- Hybrid session integration
- Manual placement analysis
- Quality metrics calculation

### Web Visualization
```bash
streamlit run session_creation_viewer_web_app.py
```
Explore generated sessions with interactive similarity analysis.

## 📊 Key Features

### Advanced Clustering Algorithm
- **Hierarchical clustering** with automatic cluster merging
- **Hybrid session support** with pre-assigned presentations
- **Size constraints** for minimum/maximum session sizes
- **Quality optimization** based on similarity metrics

### Quality Metrics
- **Session Coherence**: Average similarity within sessions
- **Session Distinctiveness**: Uniqueness compared to other sessions  
- **Presentation-Session Fit**: How well presentations match their assigned session
- **Silhouette Analysis**: Overall clustering quality assessment

### Multi-Model Embedding Support for Similarity Analysis
- **SentenceTransformers**: all-MiniLM-L6-v2, all-mpnet-base-v2, paraphrase-MiniLM-L6-v2
- **Specialized Models**: cde-small-v1/v2 for academic content
- **Custom Models**: Support for any HuggingFace compatible model

### Content Generation
- **Session Titles**: AI-generated descriptive titles
- **Keywords**: Relevant topic keywords for each session
- **Multiple LLM Options**: Ollama, local LLaMA, or Gemini API

### Data Security
- **Encryption Support**: Sensitive data encrypted with cryptpandas
- **Abstract Removal**: Generate public datasets without sensitive content
- **Password Protection**: Secure access to full datasets

## 🎯 Use Cases

### Academic Conferences
- Automatically organize submitted abstracts into thematic sessions
- Integrate invited speaker presentations into appropriate sessions
- Generate meaningful session titles and keywords
- Match sessions to relevant review committees

### Workshop Organization
- Cluster presentations by topic similarity
- Ensure balanced session sizes
- Identify potential session chairs based on expertise
- Create coherent presentation sequences

### Research Analysis
- Analyze topic clustering in academic fields
- Compare algorithmic vs. manual organization approaches
- Study presentation similarity patterns
- Evaluate clustering quality across different domains

## 📈 Quality Assurance

The system includes comprehensive quality metrics to evaluate session organization:

- **Coherence Analysis**: Measure topical focus within sessions
- **Distinctiveness Scoring**: Ensure sessions cover different topics
- **Fit Assessment**: Evaluate individual presentation placement
- **Comparative Analysis**: Compare algorithmic vs. manual approaches

## 🤝 Contributing

This package was developed for the ASABE (American Society of Agricultural and Biological Engineers) conference organization but is designed to be generalizable to other academic conferences and events.

## 📄 License

This project is developed for academic and conference organization purposes. Please ensure appropriate attribution when using or modifying the code.

## 📞 Support

For questions, issues, or feature requests, please refer to the documentation in the Jupyter notebook or examine the example workflows provided in the repository.

---

*This package represents a comprehensive solution for automated technical session creation, combining state-of-the-art NLP techniques with practical conference organization needs.*
