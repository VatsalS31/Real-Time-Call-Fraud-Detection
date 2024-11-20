
# Real-Time Fraud Call Detection with Neural Network

## Overview

This project implements a **real-time fraud call detection system** utilizing **BERT (uncased)** for feature extraction and a **transformer-based fully connected neural network**. The system is capable of detecting fraudulent or abusive calls with an accuracy of **79%**, employing **mixed precision training, regularization**, and **cross-entropy loss**.

Additionally, the project supports **multi-language live transcription**, including Indian languages, powered by **Chrome's Webkit SpeechRecognition API**. A web-based interface built using **Python Flask**, **HTML**, **CSS**, and **JavaScript** ensures a seamless and efficient user experience.

## Features

- **Fraud Detection**: Leverages BERT uncased and neural networks for real-time call fraud detection.
- **Live Transcription**: Converts audio to text in real-time with multi-language support.
- **Web Interface**: User-friendly web-based interface for real-time transcription and fraud detection.
- **Efficiency**: Achieved 79% accuracy with performance optimizations like mixed precision training.

## File Structure

```plaintext
.
├── app.py                     # Main Flask application file
├── bert_based_uncased         # Model weights and configurations for BERT uncased
├── fraud_calls_data.csv       # Dataset for training and testing the model
├── model_training.py          # Script for training the neural network
├── modeltest.html             # HTML template for the web interface
├── test.py                    # Testing script for the trained model
```

## Getting Started

### Prerequisites

- Python 3.8 or higher
- Flask
- PyTorch
- Transformers (Hugging Face)
- Pandas
- NumPy

### Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/real-time-fraud-detection.git
   cd real-time-fraud-detection
   ```

2. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Download and save pre-trained BERT weights in the `bert_based_uncased` directory.

### Dataset

The dataset (`fraud_calls_data.csv`) includes text transcriptions of calls labeled as fraudulent or non-fraudulent. You can replace it with your own dataset for custom training.

### Training the Model

Train the model using the provided script:
```bash
python model_training.py
```
This will save the trained model in the `bert_based_uncased` directory.

### Running the Application

1. Start the Flask server:
   ```bash
   python app.py
   ```

2. Open your browser and navigate to `http://localhost:5000` to use the web interface.

## Usage

- Upload or live-transcribe audio calls using the web interface.
- The system will display the transcribed text and highlight fraud or abusive content detected.

## Demo

Include screenshots or a GIF showing the application's interface and functionality.
<img width="1331" alt="image" src="https://github.com/user-attachments/assets/53ee48be-904f-49f2-a76c-e9480ddf0e44">
<img width="1334" alt="image" src="https://github.com/user-attachments/assets/b5877c70-6288-4ac8-b0b5-44c7f3ce363d">


## Contributing

Feel free to fork this repository and submit pull requests for improvements or new features.

## License

This project is licensed under the [MIT License](LICENSE).
