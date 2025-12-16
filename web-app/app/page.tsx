"use client";

import { useState } from "react";

export default function Home() {
  const [model, setModel] = useState<"2022" | "2025">("2025");
  const [inputMode, setInputMode] = useState<"text" | "url">("text");
  const [text, setText] = useState("");
  const [url, setUrl] = useState("");
  const [apiKey, setApiKey] = useState("");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<any>(null);
  const [error, setError] = useState("");

  const handleAnalyze = async () => {
    setLoading(true);
    setError("");
    setResult(null);

    let textToAnalyze = text;

    try {
      // Step 1: Scrape if URL mode
      if (inputMode === "url") {
        if (!url) throw new Error("Please enter a URL.");

        const scrapeRes = await fetch("http://localhost:8000/scrape", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ url, api_key: apiKey }),
        });

        if (!scrapeRes.ok) {
          const err = await scrapeRes.json();
          throw new Error(err.detail || "Failed to scrape URL.");
        }

        const scrapeData = await scrapeRes.json();
        textToAnalyze = scrapeData.content;
      }

      if (!textToAnalyze.trim()) {
        throw new Error("No text to analyze.");
      }

      // Step 2: Predict
      const endpoint = model === "2025" ? "/predict/2025" : "/predict/2022";
      const predictRes = await fetch(`http://localhost:8000${endpoint}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: textToAnalyze }),
      });

      if (!predictRes.ok) {
        const err = await predictRes.json();
        throw new Error(err.detail || "Prediction failed.");
      }

      const data = await predictRes.json();
      setResult(data);
    } catch (err: any) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gray-50 text-gray-900 font-sans p-8">
      <main className="max-w-3xl mx-auto bg-white rounded-xl shadow-lg overflow-hidden">
        {/* Header */}
        <div className="bg-blue-600 p-6 text-white">
          <h1 className="text-3xl font-bold">Fake News Detector</h1>
          <p className="opacity-90 mt-2">
            Analyze news articles using advanced AI models.
          </p>
        </div>

        <div className="p-6 space-y-6">
          {/* Model Selection */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Select Model
            </label>
            <div className="flex space-x-4">
              <button
                onClick={() => setModel("2025")}
                className={`flex-1 py-3 px-4 rounded-lg border-2 transition-all ${
                  model === "2025"
                    ? "border-blue-600 bg-blue-50 text-blue-700 font-bold"
                    : "border-gray-200 hover:border-gray-300"
                }`}
              >
                <div className="text-lg">2025 Model</div>
                <div className="text-xs opacity-75 font-normal">
                  Transformer (DistilRoBERTa)
                </div>
              </button>
              <button
                onClick={() => setModel("2022")}
                className={`flex-1 py-3 px-4 rounded-lg border-2 transition-all ${
                  model === "2022"
                    ? "border-blue-600 bg-blue-50 text-blue-700 font-bold"
                    : "border-gray-200 hover:border-gray-300"
                }`}
              >
                <div className="text-lg">2022 Model</div>
                <div className="text-xs opacity-75 font-normal">
                  Classic ML (Logistic Regression)
                </div>
              </button>
            </div>
          </div>

          {/* Input Mode */}
          <div>
            <div className="flex border-b border-gray-200 mb-4">
              <button
                onClick={() => setInputMode("text")}
                className={`pb-2 px-4 font-medium ${
                  inputMode === "text"
                    ? "text-blue-600 border-b-2 border-blue-600"
                    : "text-gray-500 hover:text-gray-700"
                }`}
              >
                Paste Text
              </button>
              <button
                onClick={() => setInputMode("url")}
                className={`pb-2 px-4 font-medium ${
                  inputMode === "url"
                    ? "text-blue-600 border-b-2 border-blue-600"
                    : "text-gray-500 hover:text-gray-700"
                }`}
              >
                Analyze URL
              </button>
            </div>

            {inputMode === "text" ? (
              <textarea
                className="w-full h-48 p-4 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                placeholder="Paste the news article text here..."
                value={text}
                onChange={(e) => setText(e.target.value)}
              />
            ) : (
              <div className="space-y-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    Article URL
                  </label>
                  <input
                    type="url"
                    className="w-full p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500"
                    placeholder="https://example.com/news/article"
                    value={url}
                    onChange={(e) => setUrl(e.target.value)}
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    Firecrawl API Key (Optional if set on server)
                  </label>
                  <input
                    type="password"
                    className="w-full p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500"
                    placeholder="fc-..."
                    value={apiKey}
                    onChange={(e) => setApiKey(e.target.value)}
                  />
                  <p className="text-xs text-gray-500 mt-1">
                    Required for scraping. Get one at firecrawl.dev
                  </p>
                </div>
              </div>
            )}
          </div>

          {/* Action Button */}
          <button
            onClick={handleAnalyze}
            disabled={loading}
            className={`w-full py-4 rounded-lg text-white font-bold text-lg transition-all ${
              loading
                ? "bg-gray-400 cursor-not-allowed"
                : "bg-blue-600 hover:bg-blue-700 shadow-md hover:shadow-lg"
            }`}
          >
            {loading ? "Analyzing..." : "Analyze Veracity"}
          </button>

          {/* Error Message */}
          {error && (
            <div className="p-4 bg-red-50 text-red-700 rounded-lg border border-red-200">
              <strong>Error:</strong> {error}
            </div>
          )}

          {/* Results */}
          {result && (
            <div className="mt-8 border-t pt-6 animate-in fade-in slide-in-from-bottom-4 duration-500">
              <h2 className="text-xl font-bold mb-4">Analysis Result</h2>

              <div
                className={`p-6 rounded-xl border-l-8 ${
                  result.verdict === "Reliable"
                    ? "bg-green-50 border-green-500 text-green-900"
                    : result.verdict === "Fake"
                    ? "bg-red-50 border-red-500 text-red-900"
                    : "bg-yellow-50 border-yellow-500 text-yellow-900"
                }`}
              >
                <div className="flex justify-between items-start">
                  <div>
                    <div className="text-sm font-semibold uppercase tracking-wide opacity-75">
                      Verdict
                    </div>
                    <div className="text-4xl font-extrabold mt-1">
                      {result.verdict}
                    </div>
                  </div>
                  <div className="text-right">
                    <div className="text-sm font-semibold uppercase tracking-wide opacity-75">
                      Confidence
                    </div>
                    <div className="text-3xl font-bold mt-1">
                      {(result.confidence * 100).toFixed(1)}%
                    </div>
                  </div>
                </div>

                <div className="mt-4 pt-4 border-t border-black/10 flex justify-between text-sm">
                  <span>
                    Model Used: <strong>{result.model}</strong>
                  </span>
                  <span>Label: {result.label}</span>
                </div>
              </div>
            </div>
          )}
        </div>
      </main>
    </div>
  );
}
