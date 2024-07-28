import { OpenAIEmbeddings } from "@langchain/openai";
import { PGVectorStore } from "@langchain/community/vectorstores/pgvector";
import fs from 'fs/promises';

const embeddings = new OpenAIEmbeddings({
  openAIApiKey: process.env.OPENAI_API_KEY,
  batchSize: 512,
  modelName: "text-embedding-3-large",
  // Vectors with up to 2,000 dimensions can be indexed
  // https://github.com/pgvector/pgvector#hnsw
  dimensions: 2000,
});

// Get the query string from command-line arguments
var queryString = process.argv[2];

if (!queryString) {
  console.warn("If you don't provide a query string as a command-line argument. We will use the default query string: 'frontend dev'");
  queryString = "frontend dev";
}

const pgvectorConfig = {
  postgresConnectionOptions: {
    type: "postgres",
    host: "127.0.0.1",
    port: 5432,
    database: "vector_demo",
  },
  tableName: "similarity_search",
  columns: {
    idColumnName: "id",
    vectorColumnName: "vector",
    contentColumnName: "content",
    metadataColumnName: "metadata",
  },
};

const pgvectorStore = await PGVectorStore.initialize(
  embeddings,
  pgvectorConfig
);

await pgvectorStore.client.query("DELETE FROM similarity_search;");

console.log("Cleaned database!");

try {
  // Read the JSON file
  const data = await fs.readFile('similarity_search/data.json', 'utf-8');
  
  // Parse the JSON data
  const documents = JSON.parse(data);
  
  // Add documents to pgvectorStore
  await pgvectorStore.addDocuments(documents);
  
  console.log('Documents added successfully');
} catch (error) {
  console.error('Error adding documents:', error);
  const files = await fs.readdir('./');
  console.log('Current directory files:', files);
}

// Search using cosine distance by default
// https://github.com/langchain-ai/langchainjs/blob/5df74e3/libs/langchain-community/src/vectorstores/pgvector.ts#L423

const results = await pgvectorStore.similaritySearchWithScore(queryString, 10);

console.log(`The nearest neighbors of "${queryString}" by cosine distance are:`);
console.log(results);

// await pgvectorStore.client.end();
await pgvectorStore.end();
