export default function ModelSelector({ model, setModel }) {
  return (
    <select value={model} onChange={(e) => setModel(e.target.value)}>
      <option value="svm">TF-IDF + SVM</option>
      <option value="lstm">LSTM</option>
      <option value="transformer">Transformer</option>
    </select>
  );
}