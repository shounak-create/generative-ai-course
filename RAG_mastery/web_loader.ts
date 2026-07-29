import { CheerioWebBaseLoader } from "@langchain/community/document_loaders/web/cheerio";

async function testWebLoader() {
  console.log("🌐 Fetching web page...");

  // 1. Initialize the loader with a target URL
  const loader = new CheerioWebBaseLoader(
    "https://js.langchain.com/docs/get_started/introduction"
  );

  // 2. Scrape and parse HTML into LangChain Document objects
  const docs = await loader.load();

  // 3. Inspect the loaded document structure
  console.log(`\n✅ Loaded ${docs.length} document(s).`);
  console.log("-----------------------------------------");
  console.log("Sample Text (First 300 chars):");
  console.log(docs[0].pageContent.slice(0, 300) + "...\n");
  console.log("Metadata:", docs[0].metadata);
}

testWebLoader().catch(console.error);