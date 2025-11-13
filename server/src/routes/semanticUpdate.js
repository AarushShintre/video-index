// server/src/routes/semanticUpdate.js

import express from "express";
import { updateSemanticIndex, checkSemanticSearchHealth } from "../controllers/semanticSearchController.js";

const router = express.Router();

router.get("/health", checkSemanticSearchHealth);
router.post("/update-index", updateSemanticIndex);

export default router;
