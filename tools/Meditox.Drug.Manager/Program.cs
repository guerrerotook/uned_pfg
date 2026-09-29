using HtmlAgilityPack;
using Meditox.Drug.Manager.Model;
using System.Text;
using System.Web;

namespace Meditox.Drug.Manager;

public class Program
{
    static void Main(string[] args)
    {
        string rootPath = "/Users/guerrerotook/Git/uned_pfg/data/medicamentos/";
        FileManager fileManager = new FileManager(rootPath);
        List<Model.Drug> allDrugs = fileManager.GetAllTechnicalDrugInformation();
        var drugWith48Section = allDrugs
            .Where(p => p.TechnicalDrugInfos.Any(i => i.Section == "4.8"))
            .Select(p => p);

        List<string> content = new List<string>();
        var converter = new ReverseMarkdown.Converter(new ReverseMarkdown.Config()
        {
            SuppressDivNewlines = true,
            RemoveComments = true,
            GithubFlavored = true,
        });
        string templateDrugInfo = "El medicamento {0} con los principios activos {1} tiene la siguiente información en la sección 4.8";
        foreach (var drug in drugWith48Section)
        {            
        // Parallel.ForEach(drugWith48Section, drug =>
        // {
            TechnicalDrugInfo section48 = drug.TechnicalDrugInfos
                .Where(i => i.Section == "4.8")
                .First();
            if (section48.Conent != null)
            {
                StringBuilder sb = new StringBuilder();
                sb.AppendFormat(templateDrugInfo, drug.nombre, drug.pactivos);
                sb.AppendLine();
                sb.AppendLine(section48.Conent);


                HtmlDocument doc = new HtmlDocument();
                doc.LoadHtml(converter.Convert(sb.ToString()));
                File.WriteAllText(
                    Path.Combine(rootPath, "txt", $"{drug.nregistro}.txt"),
                    HttpUtility.HtmlDecode(
                        SanitizeText(doc.DocumentNode.InnerText)));
            }
        }
        //});
    }

    private static string SanitizeText(string value)
    {
        value = value.Replace("|", string.Empty);
        value = value.Replace("---", string.Empty);
        value = RemoveAdverseReactions(value);
        return value;
    }

    private static string RemoveAdverseReactions(string value)
    {
        string result = value;
        string[] content = value.Split(Environment.NewLine);
        string? target = content
            .Where(static p => p.ToLowerInvariant().Contains("Notificación de sospechas de reacciones adversas".ToLowerInvariant()))
            .FirstOrDefault();
        if (target != null)
        {
            int index = content.ToList().IndexOf(target);
            var cleanMarkdown = content.Take(new Range(0, index));
            result = string.Join(Environment.NewLine, cleanMarkdown);
        }

        return result;
    }
}
