# ⚡ AvaTarArTs Suite

**Unified AI-Powered Creative & Development Platform**

> A comprehensive suite consolidating 12+ repositories into one powerful, organized system for AI-powered content creation, automation, and development.

---

## 🎯 Mission

Leverage 98+ AI APIs to create, automate, and innovate across:
- 🎨 **Creative Content** (Images, Audio, Video)
- 🤖 **AI-Powered Analysis**
- ⚙️ **Automation & Workflows**
- 🛠️ **Development Tools**
- 📊 **Data Management**

---

## 🧠 Powered By

### LLM & AI
- OpenAI (GPT-5) | Anthropic (Claude) | Groq | XAI (Grok)
- Deepseek | Mistral | Perplexity | Together AI

### Image Generation
- Leonardo AI | Stability AI | Replicate | FAL AI
- Remove.bg | VanceAI | Imagga

### Audio & Voice
- ElevenLabs | Deepgram | AssemblyAI
- Murf | Resemble | Suno (Music)

### Video
- Runway ML | Stable Video Diffusion | HeyGen

### Automation & Infrastructure
- n8n | Make | Zapier
- Supabase | Cloudflare R2
- Pinecone | Qdrant | ChromaDB

---

## 📦 Structure

```
AvaTarArTs-Suite/
├── 🧠 core/                    # Shared libraries & AI analysis
│   ├── shared_libs/            # Common utilities, quality tools
│   └── ai_analysis/            # AI-powered code analysis
│
├── 🎬 media/                   # Media processing tools
│   ├── audio/                  # Audio tools (70+ scripts)
│   ├── image/                  # Image processing (300+ scripts)
│   ├── video/                  # Video tools (120+ scripts)
│   └── processing/             # Format conversion
│
├── 🤖 automation/              # Automation platforms
│   ├── api_integrations/       # API wrappers (65+ integrations)
│   ├── social_media/           # Instagram, Reddit, Twitter bots
│   ├── web/                    # Web scraping & automation
│   └── youtube/                # YouTube automation tools
│
├── ✍️ content/                 # Content creation tools
│
├── 💾 data/                    # Data management & organization
│
├── 🛠️ devtools/                # Development utilities
│   ├── analysis_tools/         # Code analysis
│   ├── development_utilities/  # Dev helpers
│   └── documentation_tools/    # Doc generators
│
├── 🔧 utilities/               # System utilities
│   ├── batch/                  # Batch processors
│   ├── duplicates/             # Duplicate file finders
│   ├── organizers/             # File organization tools
│   └── system/                 # System cleanup tools
│
├── 🧪 experimental/            # Experimental projects
│
├── 📦 archived/                # Archived projects & media
│   ├── archive_files/          # Zipped projects
│   └── media_files/            # Reference media
│
└── 📚 docs/                    # Documentation
    ├── guides/                 # How-to guides
    ├── reports/                # Analysis reports
    └── organization/           # Organization docs
```

---

## 🚀 Quick Start

### 1. Load API Keys
```bash
# Load all your API keys from ~/.env.d/
source ~/.env.d/loader.sh

# Verify keys are loaded
echo $LEONARDO_API_KEY
echo $OPENAI_API_KEY
```

### 2. Explore the Suite
```bash
cd ~/GitHub/AvaTarArTs-Suite

# Browse media processing tools
ls media/audio/
ls media/image/
ls media/video/

# Check automation scripts
ls automation/api_integrations/
ls automation/social_media/

# View utilities
ls utilities/
```

### 3. Run Example Scripts
```bash
# Image processing
python3 media/image/[your_script].py

# Audio processing
python3 media/audio/[your_script].py

# Automation
python3 automation/social_media/[your_script].py
```

---

## 🎓 Key Features

### 🎨 Creative Suite
- **Image Generation**: Leonardo AI, Stability AI, Replicate integrations
- **Image Enhancement**: VanceAI upscaling, Remove.bg background removal
- **Image Analysis**: Imagga tagging and categorization
- **Audio/Voice**: ElevenLabs voice cloning, Deepgram transcription
- **Music**: Suno music generation
- **Video**: Runway ML, Stable Video Diffusion

### 🤖 AI Analysis
- Content-aware code analysis
- Quality improvement tools
- Automated testing frameworks
- Documentation generators

### ⚙️ Automation
- **65+ API Integrations**: Pre-built wrappers for popular services
- **Social Media Bots**: Instagram, Reddit, Twitter automation
- **Web Scraping**: Advanced scraping tools
- **YouTube**: Playlist management, download automation
- **n8n Workflows**: Ready-to-use workflow templates

### 🛠️ Development Tools
- Batch processors for bulk operations
- Duplicate file finders
- Advanced file organization
- System cleanup utilities

---

## 📊 Statistics

- **1,493+** Python scripts
- **98+** API integrations configured
- **12** repositories consolidated
- **70+** audio processing tools
- **300+** image processing scripts
- **120+** video tools
- **65+** API integration wrappers

---

## 🔑 Environment Setup

This suite integrates with your `~/.env.d/` API key management system:

```bash
# Load ALL APIs
source ~/.env.d/loader.sh

# Load specific categories
source ~/.env.d/loader.sh llm-apis art-vision

# Check what's loaded
env | grep API_KEY | cut -d= -f1 | sort
```

**Configured Services:**
- LLM APIs (9 services)
- Art/Vision APIs (10 services)
- Audio/Music APIs (9 services)
- Automation APIs (4 services)
- Cloud Infrastructure (3 services)
- Vector/Memory DBs (5 services)
- And more...

---

## 🛡️ Security

- ✅ All API keys stored securely in `~/.env.d/` (600 permissions)
- ✅ No hardcoded credentials in code
- ✅ `.gitignore` configured to exclude sensitive files
- ✅ Automatic backups of environment configuration

---

## 📖 Documentation

### Quick References
- `docs/guides/` - Step-by-step tutorials
- `docs/reports/` - Analysis and scan results
- `docs/organization/` - Organization documentation

### Key Docs
- [Comprehensive Analysis Summary](docs/COMPREHENSIVE_ANALYSIS_SUMMARY.md)
- [Improvement Plan](docs/COMPREHENSIVE_IMPROVEMENT_PLAN.md)
- [Quality Improvements](docs/COMPREHENSIVE_IMPROVEMENTS_IMPLEMENTED.md)

---

## 🔄 Workflow Examples

### Example 1: AI-Powered Image Generation
```bash
source ~/.env.d/loader.sh art-vision

# Generate image with Leonardo AI
python3 automation/api_integrations/leonardo_generate.py \
  --prompt "futuristic cityscape" \
  --output ./outputs/

# Enhance with VanceAI
python3 media/image/vanceai_upscale.py \
  --input ./outputs/image.png \
  --scale 4x
```

### Example 2: Content Creation Pipeline
```bash
source ~/.env.d/loader.sh

# 1. Generate script with GPT-5
# 2. Create voiceover with ElevenLabs
# 3. Generate images with Leonardo AI
# 4. Create video with Runway ML
# 5. Upload to Cloudflare R2
```

### Example 3: File Organization
```bash
# Find duplicates
python3 utilities/duplicates/fdupes/sorts.py

# Organize by type
python3 utilities/organizers/file_sorter/organize.py

# Batch process
python3 utilities/batch/batch_processor.py
```

---

## 🤝 Contributing

This is a personal toolkit consolidation, but contributions welcome:

1. Add new API integrations to `automation/api_integrations/`
2. Create new tools in appropriate directories
3. Update documentation in `docs/`
4. Submit PRs with clear descriptions

---

## 📝 License

Personal use. API keys and credentials are user-specific.

---

## 🔗 Related Projects

- [GitHub Audit Tool](../github-audit-bundle/) - Repo management
- [Environment Manager](~/.env.d/) - API key management

---

## ✨ What's Next?

### Planned Features
- [ ] Unified CLI tool for common operations
- [ ] n8n workflow templates
- [ ] Docker containerization
- [ ] Web dashboard for monitoring
- [ ] API usage tracking
- [ ] Automated testing suite
- [ ] Example project templates

### Integration Goals
- [ ] Supabase backend for content storage
- [ ] Pinecone vector search for content discovery
- [ ] Cloudflare R2 for media hosting
- [ ] n8n for workflow orchestration

---

**Built with ❤️ by ichoake**

*Last Updated: November 1, 2025*

---

## 🆘 Support

For issues or questions:
1. Check `docs/` for documentation
2. Review example scripts in each directory
3. Consult API documentation links in `~/.env.d/README.md`

**API Documentation:**
- [OpenAI](https://platform.openai.com/docs)
- [Anthropic](https://docs.anthropic.com/)
- [Leonardo AI](https://docs.leonardo.ai/)
- [ElevenLabs](https://docs.elevenlabs.io/)
- [Replicate](https://replicate.com/docs)

---

🚀 **Happy Building!**
