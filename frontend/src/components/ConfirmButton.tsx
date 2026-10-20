import { useEffect, useState } from "react";

/** Zwei-Klick-Bestätigung statt Browser-Dialog. */
export function ConfirmButton({
  onConfirm,
  label = "Löschen",
  confirmLabel = "Wirklich löschen?",
}: {
  onConfirm: () => void;
  label?: string;
  confirmLabel?: string;
}) {
  const [armed, setArmed] = useState(false);

  useEffect(() => {
    if (!armed) return;
    const timer = setTimeout(() => setArmed(false), 3000);
    return () => clearTimeout(timer);
  }, [armed]);

  return (
    <button
      type="button"
      className={armed ? "danger" : "ghost"}
      onClick={(e) => {
        e.preventDefault();
        e.stopPropagation();
        if (armed) onConfirm();
        setArmed(!armed);
      }}
    >
      {armed ? confirmLabel : label}
    </button>
  );
}
