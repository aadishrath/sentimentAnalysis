import GaugeChart from "react-gauge-chart";

export default function SentimentGauge({ score }) {
  const percent = (score - 1) / 4;
  const labels = ["Very Negative", "Negative", "Neutral", "Positive", "Very Positive"];

  return (
    <div style={{ maxWidth: "400px", margin: "auto" }}>
      <GaugeChart
        id="sentiment-gauge"
        nrOfLevels={5}
        percent={percent}
        textColor="#000"
        arcWidth={0.3}
        colors={["#d9534f", "#f0ad4e", "#f7f7f7", "#5bc0de", "#5cb85c"]}
        formatTextValue={() => `Score: ${score}`}
      />

      <p style={{ textAlign: "center", marginTop: "0.5rem" }}>
        {labels[score - 1]}
      </p>

      </div>
  );
}
