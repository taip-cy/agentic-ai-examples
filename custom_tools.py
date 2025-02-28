from smolagents.tools import Tool
import requests

class NVDCveDetailsLookupTool(Tool):
    name = "nvd_cve_details_lookup"
    description = "Retrieves vulnerability data from the NVD API for a given CVE ID."
    inputs = {
        "cve_id": {
            "type": "string",
            "description": "The CVE ID to lookup, e.g., 'CVE-2021-34527'."
        }
    }
    output_type = "string"

    def forward(self, cve_id: str) -> str:
        url = f"https://services.nvd.nist.gov/rest/json/cves/2.0?cveId={cve_id}"
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            data = response.json()
            return f"Data for {cve_id}:\n{data}"
        except Exception as e:
            return f"Error retrieving data for {cve_id}: {e}"
