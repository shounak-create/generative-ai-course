import { PDFLoader } from "@langchain/community/document_loaders/fs/pdf";
import * as path from "path";

async function testPdfLoader() {
  console.log("📄 Loading PDF document...");

  // Path to a sample PDF file in your directory
  const pdfPath = path.join(__dirname, "sample.pdf");

  // Initialize PDFLoader
  // Note: splitPages: true (default) creates 1 Document object per page
  const loader = new PDFLoader(pdfPath, {
    splitPages: true,
  });

  const docs = await loader.load();

  console.log(`\n✅ Parsed ${docs.length} page(s) from PDF.`);
  console.log("-----------------------------------------");

  if (docs.length > 0) {
    console.log("Page 1 Sample Text (First 300 chars):");
    console.log(docs[0].pageContent.slice(0, 300) + "...\n");
    
    // Inspect automatic page metadata added by LangChain
    console.log("Page 1 Metadata:", docs[0].metadata);
  }
}

testPdfLoader().catch((err) => {
  console.error("❌ Failed to parse PDF:", err.message);
  console.log("💡 Tip: Ensure a 'sample.pdf' file exists in your project folder.");
});