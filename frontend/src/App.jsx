import React, { useState } from "react";
import axios from "axios";
import ModelSelector from "./components/ModelSelector";
import InputBox from "./components/InputBox";
import SentimentGauge from "./components/SentimentGauge";

// Emoji feedback mapping
const emojiMap = {
  1: "😠",
  2: "😕",
  3: "😐",
  4: "🙂",
  5: "😄"
};

function App() {
  const [model, setModel] = useState("svm");
  const [text, setText] = useState("");
  const [score, setScore] = useState(null);
  const [confidence, setConfidence] = useState(null);

  // Submit input to backend and receive prediction
  const handleSubmit = async () => {
    try {
      const res = await axios.post("http://localhost:5000/predict", {
        model,
        text
      });
      setScore(res.data.sentiment);
      setConfidence(res.data.confidence);
    } catch (err) {
      console.error("Prediction failed", err);
    }
  };

  return (
    <div style={{ padding: "2rem", maxWidth: "600px", margin: "auto" }}>
      <h2>Sentiment Analyzer</h2>
      <ModelSelector model={model} setModel={setModel} />
      <InputBox text={text} setText={setText} />
      <button onClick={handleSubmit} style={{ marginTop: "1rem" }}>
        Analyze
      </button>

      {score && (
        <div style={{ marginTop: "2rem" }}>
          <SentimentGauge score={score} />
          <p>Confidence: {(confidence * 100).toFixed(2)}%</p>
          <p>Emoji Feedback: {emojiMap[score]}</p>
        </div>
      )}
    </div>
  );
}

export default App;
