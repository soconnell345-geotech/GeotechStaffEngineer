# DIGGS 2.6 schema (bundled)

`diggs-schema-2.6.zip` holds the DIGGS 2.6 XML schema files, **unmodified**:
- `Diggs.xsd`;
- every file it imports or includes;
- 30 files in all, with their relative paths kept.

**Source.** DIGGSml (`https://github.com/DIGGSml`, the DIGGS 2.6 production release), as redistributed in the `schemas/diggs-schema-2.6/` folder of pydiggs 0.1.5.

**License.** The schema is published under the **Mozilla Public License 2.0** (`LICENSE-DIGGS-SCHEMA`, beside this file). The files are redistributed unchanged.

**Why it is bundled.** It is the only route to a DIGGS schema check that works on every host:
- pydiggs is an optional extra, deliberately not installed on Databricks (since 5.10.1, its dependencies replaced the notebook's Pygments and Databricks killed the kernel);
- the schema check needs nothing but lxml, which is already a dependency;
- nothing is fetched from the network.

`subsurface_characterization.formats.diggs_validation` unpacks the zip once into a cache folder on first use.

**Updating to a newer DIGGS release:**
1. Rebuild the zip from `Diggs.xsd` by walking its `xs:import` / `xs:include` locations.
2. Keep the file names and relative paths.
3. Update this note and the version constant in that module.
