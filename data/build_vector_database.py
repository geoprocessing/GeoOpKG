# Script to import from Neo4j to Weaviate
import weaviate
import weaviate.classes as wvc
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer
import os

# ===========================
# Configuration - please edit here!
# ===========================
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = ""
NEO4J_PASSWORD = ""  # ⚠️ Set your Neo4j password

WEAVIATE_HOST = "localhost"
WEAVIATE_PORT = 8080
WEAVIATE_GRPC_PORT = 50051

# Vector model
print("🔄 Loading vector model...")
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
print("✅ Vector model loaded")


# ===========================
# Step 1: Delete old data (optional)
# ===========================
def clear_old_collections():
	"""Delete test data and rebuild collections"""
	client = weaviate.connect_to_local(
		host=WEAVIATE_HOST,
		port=WEAVIATE_PORT,
		grpc_port=WEAVIATE_GRPC_PORT,
	)

	try:
		# Delete old collections
		if client.collections.exists("VectorData"):
			client.collections.delete("VectorData")
			print("✅ Deleted old VectorData collection")

		if client.collections.exists("Prompt_Desc"):
			client.collections.delete("Prompt_Desc")
			print("✅ Deleted old Prompt_Desc collection")

		# Recreate collections (fix latest API warnings)
		client.collections.create(
			name="VectorData",
			properties=[
				wvc.config.Property(name="name", data_type=wvc.config.DataType.TEXT),
				wvc.config.Property(name="label", data_type=wvc.config.DataType.TEXT),
			]
		)
		print("✅ Recreated VectorData collection")

		client.collections.create(
			name="Prompt_Desc",
			properties=[
				wvc.config.Property(name="name", data_type=wvc.config.DataType.TEXT),
				wvc.config.Property(name="description", data_type=wvc.config.DataType.TEXT),
			]
		)
		print("✅ Recreated Prompt_Desc collection")

	except Exception as e:
		print(f"❌ Error while clearing collections: {e}")
		raise
	finally:
		client.close()


# ===========================
# Step 2: Test Neo4j connection
# ===========================
def test_neo4j_connection():
	"""Test whether the Neo4j connection is working"""
	print("\n🔍 Testing Neo4j connection...")
	try:
		driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
		with driver.session() as session:
			result = session.run("RETURN 1 AS test")
			record = result.single()
			if record["test"] == 1:
				print("✅ Neo4j connection successful")

				# Count nodes
				result = session.run("MATCH (n:Operation) RETURN count(n) AS count")
				count = result.single()["count"]
				print(f"📊 Database contains {count} Operation nodes")

				driver.close()
				return True
	except Exception as e:
		print(f"❌ Neo4j connection failed: {e}")
		print("\n💡 Please check:")
		print("   1. Is the Neo4j service running?")
		print("   2. Is the password correct? (set near the top of this file)")
		print("   3. Is port 7687 occupied?")
		return False


# ===========================
# Helper: safely convert to string
# ===========================
def safe_str(value):
	"""Safely convert any type to string"""
	if value is None:
		return ""
	if isinstance(value, str):
		return value.strip()
	# Handle numeric types (int, float, etc.)
	return str(value).strip()


# ===========================
# Step 3: Fetch data from Neo4j
# ===========================
def fetch_operations_from_neo4j():
	"""Fetch operator names and descriptions from Neo4j"""
	driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
	operations = []

	try:
		with driver.session() as session:
			# Query all Operation nodes
			query = """
            MATCH (op:Operation)
            OPTIONAL MATCH (op)-[:ImplementedIn]->(soft:Software)
            RETURN 
                op.Title AS name,
                op.Name AS alt_name,
                op.Description AS description,
                soft.Name AS software,
                soft.Version AS version
            """
			result = session.run(query)

			for record in result:
				# Safely get name (may be string or other type)
				name = safe_str(record["name"]) or safe_str(record["alt_name"])
				if not name:
					continue

				operations.append({
					"name": name,
					"description": safe_str(record["description"]),
					"software": safe_str(record["software"]),
					"version": safe_str(record["version"])
				})

		print(f"📊 Fetched {len(operations)} operators from Neo4j")

		# Show first 5 examples
		if operations:
			print("\n📋 Data samples (first 5):")
			for i, op in enumerate(operations[:5], 1):
				software_info = op['software'] if op['software'] else "Unknown"
				print(f"   {i}. {op['name']} ({software_info})")

		return operations

	except Exception as e:
		print(f"❌ Failed to fetch data: {e}")
		import traceback
		traceback.print_exc()  # Print detailed error info
		raise
	finally:
		driver.close()


# ===========================
# Step 4: Build enhanced description
# ===========================
def build_enhanced_description(op):
	"""Build enriched description text for operators (for vector retrieval)"""
	parts = [f"Operation: {op['name']}"]

	if op['description']:
		parts.append(f"Description: {op['description']}")

	if op['software']:
		software_info = op['software']
		if op['version']:
			software_info += f" {op['version']}"
		parts.append(f"Software: {software_info}")

	return ". ".join(parts)


# ===========================
# Step 5: Batch import into vector database
# ===========================
def update_vector_databases(operations):
	"""Batch update two vector collections"""
	if not operations:
		print("❌ No data to import")
		return

	client = weaviate.connect_to_local(
		host=WEAVIATE_HOST,
		port=WEAVIATE_PORT,
		grpc_port=WEAVIATE_GRPC_PORT,
	)

	try:
		vector_data_col = client.collections.get("VectorData")
		prompt_desc_col = client.collections.get("Prompt_Desc")

		# Batch import VectorData (for entity disambiguation)
		print("\n🔄 Updating VectorData collection...")
		vector_data_objects = []
		for op in operations:
			# Vectorize operator name
			name_vector = model.encode(op['name']).tolist()
			vector_data_objects.append(
				wvc.data.DataObject(
					properties={
						"name": op['name'],
						"label": "Operation"
					},
					vector=name_vector
				)
			)

		# Batch insert (new API)
		response = vector_data_col.data.insert_many(vector_data_objects)
		success_count = len(operations) - len(response.errors)
		print(f"✅ VectorData: inserted {success_count}/{len(operations)} records")
		if response.errors:
			print(f"⚠️  {len(response.errors)} records failed to insert")

		# Batch import Prompt_Desc (for question matching)
		print("\n🔄 Updating Prompt_Desc collection...")
		prompt_desc_objects = []
		for op in operations:
			# Vectorize enhanced description
			enhanced_desc = build_enhanced_description(op)
			desc_vector = model.encode(enhanced_desc).tolist()
			prompt_desc_objects.append(
				wvc.data.DataObject(
					properties={
						"name": op['name'],
						"description": enhanced_desc
					},
					vector=desc_vector
				)
			)

		response = prompt_desc_col.data.insert_many(prompt_desc_objects)
		success_count = len(operations) - len(response.errors)
		print(f"✅ Prompt_Desc: inserted {success_count}/{len(operations)} records")
		if response.errors:
			print(f"⚠️  {len(response.errors)} records failed to insert")

	except Exception as e:
		print(f"❌ Failed to import into vector database: {e}")
		raise
	finally:
		client.close()


# ===========================
# Step 6: Verify data
# ===========================
def verify_data():
	"""Verify data in the vector database"""
	client = weaviate.connect_to_local(
		host=WEAVIATE_HOST,
		port=WEAVIATE_PORT,
		grpc_port=WEAVIATE_GRPC_PORT,
	)

	try:
		# Check VectorData
		vector_data = client.collections.get("VectorData")
		agg_result = vector_data.aggregate.over_all(total_count=True)
		count1 = agg_result.total_count

		# Check Prompt_Desc
		prompt_desc = client.collections.get("Prompt_Desc")
		agg_result = prompt_desc.aggregate.over_all(total_count=True)
		count2 = agg_result.total_count

		print("\n" + "=" * 60)
		print("📈 Data verification results")
		print("=" * 60)
		print(f"VectorData collection:   {count1} records (for entity disambiguation)")
		print(f"Prompt_Desc collection:  {count2} records (for question matching)")

		# Test queries
		print("\n🔍 Testing query functionality...")
		test_queries = ["Buffer", "Clip", "Raster Calculator"]

		for query_text in test_queries:
			print(f"\nQuery: '{query_text}'")
			test_vector = model.encode(query_text).tolist()

			response = vector_data.query.near_vector(
				near_vector=test_vector,
				limit=3,
				return_metadata=wvc.query.MetadataQuery(certainty=True)
			)

			if response.objects:
				for i, obj in enumerate(response.objects, 1):
					certainty = obj.metadata.certainty if obj.metadata and obj.metadata.certainty else 0
					print(f"  {i}. {obj.properties['name']} (Similarity: {certainty:.4f})")
			else:
				print("  No related results found")

		print("\n" + "=" * 60)

	except Exception as e:
		print(f"❌ Data verification failed: {e}")
		raise
	finally:
		client.close()


# ===========================
# Main Program
# ===========================
if __name__ == "__main__":
	print("=" * 60)
	print("🚀 Vector Database Update Tool")
	print("=" * 60)

	# Step 0: Test Neo4j connection
	if not test_neo4j_connection():
		print("\n❌ Program terminated: Neo4j connection failed")
		print("\nPlease update the password near the top of this file:")
		print('   NEO4J_PASSWORD = "your_password"')
		exit(1)

	# Step 1: Clear old data
	print("\n" + "-" * 60)
	choice = input("Delete old data and rebuild? (y/n): ").strip().lower()
	if choice == 'y':
		clear_old_collections()

	# Step 2: Fetch data from Neo4j
	print("\n" + "-" * 60)
	try:
		operations = fetch_operations_from_neo4j()
	except Exception as e:
		print("\n❌ Program terminated: failed to fetch data from Neo4j")
		exit(1)

	if not operations:
		print("❌ No operator data found, please check:")
		print("   1. Has data been imported into Neo4j?")
		print("   2. Do Operation nodes exist?")
		exit(1)

	# Step 3: Update vector database
	print("\n" + "-" * 60)
	try:
		update_vector_databases(operations)
	except Exception as e:
		print("\n❌ Program terminated: vector database update failed")
		exit(1)

	# Step 4: Verify
	print("\n" + "-" * 60)
	try:
		verify_data()
	except Exception as e:
		print(f"\n⚠️  Verification error: {e}")

	print("\n" + "=" * 60)
	print("✅ Vector database update completed!")
	print("=" * 60)
