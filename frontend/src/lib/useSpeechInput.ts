import { useCallback, useEffect, useRef, useState } from "react";

// Minimale Typen der Web Speech API (lib.dom typisiert SpeechRecognition nicht in allen Versionen)
interface SpeechRecognitionAlternativeLike {
  transcript: string;
}
interface SpeechRecognitionResultLike {
  isFinal: boolean;
  0: SpeechRecognitionAlternativeLike;
}
interface SpeechRecognitionEventLike {
  resultIndex: number;
  results: ArrayLike<SpeechRecognitionResultLike>;
}
interface SpeechRecognitionErrorEventLike {
  error: string;
}
interface SpeechRecognitionLike {
  lang: string;
  continuous: boolean;
  interimResults: boolean;
  onresult: ((event: SpeechRecognitionEventLike) => void) | null;
  onerror: ((event: SpeechRecognitionErrorEventLike) => void) | null;
  onend: (() => void) | null;
  start(): void;
  stop(): void;
  abort(): void;
}
type SpeechRecognitionConstructor = new () => SpeechRecognitionLike;

function recognitionClass(): SpeechRecognitionConstructor | undefined {
  if (typeof window === "undefined") return undefined;
  const w = window as unknown as {
    SpeechRecognition?: SpeechRecognitionConstructor;
    webkitSpeechRecognition?: SpeechRecognitionConstructor;
  };
  return w.SpeechRecognition ?? w.webkitSpeechRecognition;
}

const ERROR_MESSAGES: Record<string, string> = {
  "not-allowed": "Mikrofonzugriff wurde verweigert. Bitte in den Browser-Einstellungen erlauben.",
  "service-not-allowed": "Spracherkennung ist in diesem Browser nicht erlaubt.",
  "audio-capture": "Kein Mikrofon gefunden.",
  network: "Spracherkennung nicht erreichbar (Netzwerkfehler).",
  "language-not-supported": "Deutsch wird von der Spracherkennung dieses Browsers nicht unterstützt.",
};
// Kein Fehler für den Nutzer: Stille bzw. bewusst abgebrochen
const IGNORED_ERRORS = new Set(["no-speech", "aborted"]);

/**
 * Diktat über die Spracherkennung des Browsers (Chrome, Edge, Safari; nicht Firefox).
 * Fertig erkannte Sätze gehen an `onFinal`, das noch unfertige Zwischenergebnis steht in `interim`.
 */
export function useSpeechInput(onFinal: (text: string) => void) {
  const Recognition = recognitionClass();
  const [listening, setListening] = useState(false);
  const [interim, setInterim] = useState("");
  const [error, setError] = useState<string | null>(null);
  const recognition = useRef<SpeechRecognitionLike | null>(null);
  const finalHandler = useRef(onFinal);
  finalHandler.current = onFinal;

  const stop = useCallback(() => recognition.current?.stop(), []);
  // Ohne ausstehende Ergebnisse, z. B. beim Senden (sonst landen nachgelieferte Wörter im geleerten Feld)
  const cancel = useCallback(() => recognition.current?.abort(), []);

  const start = useCallback(() => {
    if (!Recognition || recognition.current) return;
    const r = new Recognition();
    r.lang = "de-DE";
    r.continuous = true;
    r.interimResults = true;
    r.onresult = (event) => {
      let pending = "";
      for (let i = event.resultIndex; i < event.results.length; i++) {
        const result = event.results[i];
        if (result.isFinal) finalHandler.current(result[0].transcript);
        else pending += result[0].transcript;
      }
      setInterim(pending);
    };
    r.onerror = (event) => {
      if (!IGNORED_ERRORS.has(event.error)) {
        setError(ERROR_MESSAGES[event.error] ?? `Spracherkennung fehlgeschlagen (${event.error}).`);
      }
    };
    r.onend = () => {
      recognition.current = null;
      setListening(false);
      setInterim("");
    };
    setError(null);
    recognition.current = r;
    try {
      r.start();
      setListening(true);
    } catch {
      recognition.current = null;
      setError("Spracherkennung konnte nicht gestartet werden.");
    }
  }, [Recognition]);

  useEffect(() => () => recognition.current?.abort(), []);

  return { supported: Recognition !== undefined, listening, interim, error, start, stop, cancel };
}
