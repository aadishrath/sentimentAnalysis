import GaugeChart from "react-gauge-chart";

export default function SentimentGauge({ score }) {
  // Normalize score from 1–5 to 0–1 for gauge
  const percent = (score - 1) / 4;

  return (
    <GaugeChart
      id="sentiment-gauge"
      nrOfLevels={5}
      percent={percent}
      textColor="#000"
      arcWidth={0.3}
      colors={["#ff0000", "#ffff00", "#00ff00"]}
      formatTextValue={() => `Score: ${score}`}
    />
  );
}