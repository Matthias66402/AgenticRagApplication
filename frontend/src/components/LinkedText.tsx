import type { ReactNode } from "react";

// Links im Fließtext des LLM: Markdown-Link [Text](URL) oder nackte URL (auch ohne Schema ab „www.“), wie im PDF
const LINK_PATTERN =
  /\[(?<label>[^\][\n]+)\]\((?<href>https?:\/\/[^\s()]+(?:\([^\s()]*\)[^\s()]*)*)\)|(?<url>\b(?:https?:\/\/|www\.)[^\s<>"\]]+)/g;
const TRAILING_PUNCTUATION = /[.,;:!?'"]+$/;

/** Satzzeichen und unpaarige Klammer am Ende gehören nicht zur URL, z. B. „(siehe https://a.de/x).“ */
function splitBareUrl(match: string): [url: string, rest: string] {
  let url = match.replace(TRAILING_PUNCTUATION, "");
  const count = (char: string) => url.split(char).length - 1;
  while (url.endsWith(")") && count(")") > count("(")) {
    url = url.slice(0, -1).replace(TRAILING_PUNCTUATION, "");
  }
  return [url, match.slice(url.length)];
}

/** Gibt Text aus und macht darin enthaltene URLs klickbar. */
export function LinkedText({ text }: { text: string }) {
  const parts: ReactNode[] = [];
  let position = 0;
  for (const match of text.matchAll(LINK_PATTERN)) {
    const { label, href, url } = match.groups!;
    parts.push(text.slice(position, match.index));
    let shown = label;
    let link = href;
    let rest = "";
    if (!label) {
      [shown, rest] = splitBareUrl(url);
      link = shown.includes("://") ? shown : `https://${shown}`;
    }
    parts.push(
      <a key={match.index} href={link} target="_blank" rel="noreferrer" className="text-link">
        {shown}
      </a>,
      rest,
    );
    position = match.index + match[0].length;
  }
  parts.push(text.slice(position));
  return <>{parts}</>;
}
