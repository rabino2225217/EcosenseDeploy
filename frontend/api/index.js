import { createRequestHandler } from "@react-router/node";
import * as build from "../build/server/index.js";

export default createRequestHandler(build, process.env);

export const config = {
  runtime: "nodejs20.x",
  regions: ["sfo1", "iad1", "sin1", "cdg1"],
  maxDuration: 30,
};

