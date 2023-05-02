import json
import csv

converted_fp = "give path"
normal_fp = "give path"

with open(converted_fp, "r") as json_file, open(normal_fp, "w") as csv_file:
    data = json.load(json_file)
    writer = csv.writer(csv_file)
    writer.writerow(["text", "summary", "type_ids", "row_ids", "col_ids"])
    for sample in data:
        src = sample.get("subtable_metadata_str", "")
        tgt = sample.get("sentence_annotations", [{"final_sentence": ""}])[0]["final_sentence"]
        type_ids = sample.get("type_ids", "")
        row_ids = sample.get("row_ids", "")
        col_ids = sample.get("col_ids", "")
    writer.writerow([src, tgt, type_ids, row_ids, col_ids])