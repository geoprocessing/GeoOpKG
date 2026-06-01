
## 1. Inputs, Outputs, and Expected Behavior

### Inputs
- A single natural-language question about GIS operators or geoprocessing tasks.

### Outputs
- A textual answer summarizing the most relevant operators, parameters, and explanations.

### Expected Behavior
The system classifies the question into one of three intents:
1. Known Operator: question names a specific operator
2. Similar Operator: question asks for alternatives or equivalents
3. Unknown Operator: question describes a task without a specific operator name

It then performs entity extraction, vector recall, graph retrieval, and LLM generation to produce the final answer.

## 2. Typical Use Cases (Tutorials)

### 2.1 Known Operator Question

**Input**
```
What parameters does the reclassify by table tool in GIS require as input?
```

**Expected Output Summary**
The answer lists key input parameters such as input raster, raster band, reclassification table, No Data handling, output data type, and output raster path.

### 2.2 Similar Operator Question

**Input**
```
Does QGIS have a tool that can replace ArcGIS tabulate area?
```

**Expected Output Summary**
The answer recommends QGIS Processing Toolbox alternatives such as SAGA Raster Classes Area for Polygons, and explains when Zonal Statistics can be used as a partial substitute.

### 2.3 Unknown Operator (Task-Based) Question

**Input**
```
How can I resample a 30 m DEM to 90 m resolution while preserving terrain structure as much as possible?
```

**Expected Output Summary**
The answer proposes suitable resampling strategies (cubic/bilinear interpolation or aggregation) and lists software-specific tools across ArcGIS, GRASS, and Whitebox.

## 3. Troubleshooting

- If intent recognition fails, verify DashScope API key and network access.
- If Neo4j queries fail, confirm bolt endpoint and credentials.
- If Weaviate returns empty results, verify the VectorData and Prompt_Desc collections are populated.

