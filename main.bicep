// main.bicep — Supplemental Azure Resources for EDGAR Multi-Agent Comparator
//
// WHAT THIS DEPLOYS (into your EXISTING resource group):
//   - Azure AI Search (if you need a fresh one; skip if already deployed)
//   - Azure Cosmos DB (NoSQL) — evaluation result store
//   - Azure Storage Account — raw EDGAR filing archive
//
// WHAT IT DOES NOT DEPLOY (already exists in your Foundry hub):
//   - Azure AI Foundry hub / project
//   - Azure OpenAI (linked to the hub)
//   - The existing AI Search instance in rg-apple-agentic-54-final
//
// USAGE:
//   az deployment group create \
//     --resource-group rg-apple-agentic-54-final \
//     --template-file main.bicep \
//     --parameters @main.parameters.json
//
// Or inline:
//   az deployment group create \
//     --resource-group rg-apple-agentic-54-final \
//     --template-file main.bicep \
//     --parameters location=eastus2 searchSkuName=basic

targetScope = 'resourceGroup'

// ── Parameters ────────────────────────────────────────────────────────────────

@description('Azure region for all resources — should match your Foundry hub region.')
param location string = resourceGroup().location

@description('Short suffix appended to resource names to ensure uniqueness.')
@maxLength(6)
param nameSuffix string = uniqueString(resourceGroup().id, 'edgar')

@description('Deploy a fresh AI Search instance? Set false if you already have one.')
param deploySearch bool = false

@description('AI Search SKU. basic supports semantic search. standard allows more indexes.')
@allowed(['free', 'basic', 'standard', 'standard2', 'standard3'])
param searchSkuName string = 'basic'

@description('Cosmos DB account throughput (RU/s). 400 is the minimum.')
@minValue(400)
param cosmosThroughput int = 400

// ── Variables ─────────────────────────────────────────────────────────────────

var suffix = take(nameSuffix, 6)
var searchName = 'edgar-search-${suffix}'
var cosmosName = 'edgar-cosmos-${suffix}'
var storagePrefix = 'edgar${suffix}'
var storageName = replace(toLower(storagePrefix), '-', '')  // Storage names: lowercase alphanumeric only
var cosmosDbName = 'edgar_evals'
var cosmosContainerName = 'eval_results'

// ── Azure AI Search (optional — skip if using the existing instance) ──────────

resource searchService 'Microsoft.Search/searchServices@2024-06-01-preview' = if (deploySearch) {
  name: searchName
  location: location
  sku: {
    name: searchSkuName
  }
  properties: {
    replicaCount: 1
    partitionCount: 1
    // Semantic ranker is included in basic+ SKUs
    semanticSearch: searchSkuName == 'free' ? 'disabled' : 'standard'
    // Public access ON — in production you would restrict to VNet
    publicNetworkAccess: 'enabled'
  }

  // LEARNING NOTE:
  // - 'free' tier: 1 index, no semantic search, no SLA (great for learning)
  // - 'basic': 15 indexes, semantic search, 99.9% SLA — good for this project
  // - 'standard': 50 indexes, higher throughput — production workloads
  // Vector search (HNSW/KNN) is available on ALL paid tiers.
}

// ── Azure Cosmos DB (NoSQL) — Evaluation Result Store ─────────────────────────
//
// LEARNING NOTE:
// Cosmos DB NoSQL API stores JSON documents with a partition key.
// We partition by 'ticker' so queries like "all NVDA eval results" are efficient
// (they hit a single physical partition rather than fan-out across all).

resource cosmosAccount 'Microsoft.DocumentDB/databaseAccounts@2024-05-15' = {
  name: cosmosName
  location: location
  kind: 'GlobalDocumentDB'
  properties: {
    consistencyPolicy: {
      // Session consistency: reads reflect writes from the same session.
      // Cheaper than Strong consistency, fine for eval dashboards.
      defaultConsistencyLevel: 'Session'
    }
    locations: [
      {
        locationName: location
        failoverPriority: 0
        isZoneRedundant: false
      }
    ]
    databaseAccountOfferType: 'Standard'
    enableFreeTier: true        // Use the free tier (5 GB, 1000 RU/s) if available
    publicNetworkAccess: 'Enabled'
    capabilities: [
      { name: 'EnableServerless' }  // Serverless: pay-per-request, no provisioned throughput cost
    ]
  }
}

resource cosmosDatabase 'Microsoft.DocumentDB/databaseAccounts/sqlDatabases@2024-05-15' = {
  parent: cosmosAccount
  name: cosmosDbName
  properties: {
    resource: {
      id: cosmosDbName
    }
  }
}

resource cosmosContainer 'Microsoft.DocumentDB/databaseAccounts/sqlDatabases/containers@2024-05-15' = {
  parent: cosmosDatabase
  name: cosmosContainerName
  properties: {
    resource: {
      id: cosmosContainerName
      partitionKey: {
        paths: ['/ticker']    // Partition by ticker for efficient per-company queries
        kind: 'Hash'
        version: 2
      }
      // TTL of -1 means documents live forever (no auto-expiry)
      defaultTtl: -1
      indexingPolicy: {
        automatic: true
        indexingMode: 'consistent'
        includedPaths: [{ path: '/*' }]
        excludedPaths: [{ path: '/_etag/?' }]
      }
    }
  }
}

// ── Azure Storage Account — EDGAR Filing Archive ──────────────────────────────
//
// LEARNING NOTE:
// We use Blob Storage (Hot tier) to persist raw EDGAR filings so we don't
// need to re-download from SEC on every run.  LRS replication is sufficient
// for this project (data is reproducible from SEC EDGAR).

resource storageAccount 'Microsoft.Storage/storageAccounts@2023-05-01' = {
  name: storageName
  location: location
  sku: {
    name: 'Standard_LRS'   // Locally Redundant Storage — cheapest option
  }
  kind: 'StorageV2'
  properties: {
    accessTier: 'Hot'
    supportsHttpsTrafficOnly: true
    minimumTlsVersion: 'TLS1_2'
    allowBlobPublicAccess: false   // Never expose raw filings publicly
  }
}

resource blobService 'Microsoft.Storage/storageAccounts/blobServices@2023-05-01' = {
  parent: storageAccount
  name: 'default'
}

resource edgarContainer 'Microsoft.Storage/storageAccounts/blobServices/containers@2023-05-01' = {
  parent: blobService
  name: 'edgar-filings'
  properties: {
    publicAccess: 'None'
  }
}

// ── Outputs (paste into .env) ─────────────────────────────────────────────────

output cosmosEndpoint string = cosmosAccount.properties.documentEndpoint
output storageConnectionString string = 'DefaultEndpointsProtocol=https;AccountName=${storageAccount.name};EndpointSuffix=${environment().suffixes.storage};AccountKey=${storageAccount.listKeys().keys[0].value}'
output searchEndpoint string = deploySearch ? 'https://${searchService.name}.search.windows.net' : '(using existing)'
output searchAdminKey string = deploySearch ? searchService.listAdminKeys().primaryKey : '(using existing)'

// Summary for terminal output after deployment
output deploymentSummary object = {
  cosmosDatabase: cosmosDbName
  cosmosContainer: cosmosContainerName
  storageContainer: 'edgar-filings'
  note: 'Copy cosmosEndpoint and storageConnectionString into your .env file'
}
