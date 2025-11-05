import React from "react";
import "../css/header.css";
import snowflakeIcon from "../assets/snowflake.svg";

export default function Header() {
  return (
    <div className="header flex items-center justify-between px-6 py-3">
      <div className="flex items-center gap-3">
        <img src={snowflakeIcon} alt="Snowflake" className="w-7 h-7" />
        <h1 className="text-xl font-semibold tracking-tight text-gray-700">
          Snowflake SAM
        </h1>
      </div>
      <div className="text-sm text-gray-400 italic">Iterative Segmentation</div>
    </div>
  );
}
