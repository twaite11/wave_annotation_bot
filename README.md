AI-Powered Surf Quality Scorer
This project leverages the Google AI API to autogenerate images of waves from text prompts and allows users to score the generated images based on their perceived surf quality. The scored images are saved for potential use in training machine learning models.

🚀 Getting Started
Prerequisites
Python 3.8+

Google AI API key 🔑

Git

Installation
Clone the repository:

Bash

git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
Install the required Python packages:

Bash

pip install -r requirements.txt
Set up your Google AI API key:
Create a file named .env in the project's root directory and add your API key:

GOOGLE_API_KEY=YOUR_API_KEY
## 🤖 Usage
### 1. Generate an Image
Run the main script and provide a prompt for the wave image you want to generate. For best results, be descriptive!

Bash

python main.py "A perfect glassy wave with a hollow barrel at sunset"
The script will display the generated image.

### 2. Score the Wave
After the image is displayed, you will be prompted to score the wave's surf quality.

1-3: Poor surf quality (e.g., choppy, small, mushy)

4-7: Average surf quality (e.g., rideable, but not ideal)

8-10: Excellent surf quality (e.g., clean, powerful, good shape)

Enter a number between 1 and 10 in the console.

### 3. Save the Data
Once scored, the image will be saved in the scored_images folder. The filename will include the prompt and the score, for example: a_perfect_glassy_wave_with_a_hollow_barrel_at_sunset_score_9.png.

This data can then be used to train a future model that could potentially automate the scoring of wave images.
