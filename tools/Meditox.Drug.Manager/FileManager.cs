namespace Meditox.Drug.Manager
{
    using System;
    using System.Collections.Generic;
    using System.Linq;
    using System.Text;
    using System.Text.Json;
    using System.Threading.Tasks;

    public class FileManager
    {
        private readonly string directoryPath;
        private const string InformationFolderName = "medicamentos_info";
        private const string TechnicalInformationFolderName = "medicamentos_tecnico";

        public FileManager(string directoryPath)
        {
            this.directoryPath = directoryPath;
        }

        public List<Model.Drug> GetAllTechnicalDrugInformation()
        {
            List<Model.Drug> result = new List<Model.Drug>();
            string drugInfoPath = Path.Combine(directoryPath, InformationFolderName);
            string[] drugsItems = Directory.GetFiles(drugInfoPath);
            // foreach (string drugItem in drugsItems)
            Parallel.ForEach(drugsItems, drugItem =>
            {
                string fileName = Path.GetFileNameWithoutExtension(drugItem);
                fileName = fileName.Replace("_info", string.Empty);
                string technicalInfoPath = Path.Combine(
                    directoryPath,
                    TechnicalInformationFolderName,
                    $"{fileName}_tecnico.json");
                if (File.Exists(technicalInfoPath))
                {
                    using FileStream fs = File.OpenRead(drugItem);
                    var drug = JsonSerializer.Deserialize<Model.Drug>(fs);
                    if (drug != null)
                    {
                        using FileStream fsTechnical = File.OpenRead(technicalInfoPath);
                        JsonDocument drugItemjsonDocument = JsonDocument.Parse(fsTechnical);
                        if (drugItemjsonDocument.RootElement.ValueKind != JsonValueKind.Object)
                        {
                            fsTechnical.Position = 0;

                            var technicalInformation = JsonSerializer.Deserialize<List<Model.TechnicalDrugInfo>>(
                               File.ReadAllText(technicalInfoPath));

                            if (technicalInformation != null)
                            {
                                drug.TechnicalDrugInfos = technicalInformation;
                            }

                            result.Add(drug);

                        }
                    }
                }
            });

            return result;
        }
    }
}
