# GymInsight 💪

Personal training analytics & AI coach. Visualize StrengthLog exports, rate your strength level, and get AI-powered analysis.

## Features

- **📊 Dashboard** — Volume trends, exercise progression, frequency, workout distribution
- **🏋️ Strength Level** — PR detection, strengthlevel.com ratings (Beginner → Elite)
- **⚖️ Muscle Balance** — Volume distribution, push/pull ratio, imbalance detection
- **🤖 AI Analysis** — Chat with Deepseek about your training (supports prompt caching)
- **📋 Program Builder** — Manual program creation + AI-generated programs with evaluation
- **⚙️ Settings** — API config, user profile, exercise mapping editor

## Quick Start

```bash
git clone https://github.com/mt37vn/gym-insight.git
cd gyminsight

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure Deepseek API
cp .env.example .env
# Edit .env with your DEEPSEEK_API_KEY

# Place your StrengthLog CSV export at data/training_data.csv

# Run
streamlit run app.py
```

## Data

Export your training data from StrengthLog app (Settings → Export Data → CSV).
Place the CSV at `data/training_data.csv`. The app will parse and cache it.

## Project Structure

```
├── app.py                 # Entry point
├── pages/                 # Streamlit multi-page app
│   ├── 1_Dashboard.py
│   ├── 2_Strength_Level.py
│   ├── 3_Muscle_Balance.py
│   ├── 4_AI_Analysis.py
│   ├── 5_Program_Builder.py
│   └── 6_Settings.py
├── src/                   # Core logic
│   ├── data_loader.py     # CSV parsing + caching
│   ├── processor.py       # Volume, PRs, progression
│   ├── strength_level.py  # strengthlevel.com ratings
│   ├── muscle_mapping.py  # Exercise → muscle groups
│   ├── balance.py         # Muscle balance analysis
│   ├── deepseek_client.py # Deepseek API client with caching
│   ├── program_builder.py # Program models + evaluation
│   └── prompts.py         # AI prompt templates
├── data/                  # Cached data + mappings
├── assets/style.css       # Custom theme
└── requirements.txt
```

## Deepseek Integration

- Configure API key in Settings or `.env`
- Model selector: `deepseek-chat` (flagship) or `deepseek-reasoner` (R1)
- Prompt caching enabled by default via `ephemeral_prompt_caching`
- Cache ratio shown in sidebar — training context is the cached prefix
