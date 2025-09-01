export default function InputBox({ text, setText }) {
  return (
    <textarea
      value={text}
      onChange={(e) => setText(e.target.value)}
      placeholder="Type your thoughts... 😊"
      rows={4}
      style={{ width: "100%", fontSize: "1rem" }}
    />
  );
}