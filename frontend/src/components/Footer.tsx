import React from "react";
import "../css/footer.css";

export default function Footer() {
  return (
    <div className="footer">
      © {new Date().getFullYear()} Snowflake SAM — powered by Segment Anything
    </div>
  );
}
