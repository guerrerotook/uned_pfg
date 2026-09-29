namespace Meditox.Drug.Manager.Model
{
    using System.Text.Json.Serialization;

    /// <summary>
    /// Represents the technical information of a drug.
    /// </summary>
    public class TechnicalDrugInfo
    {
        /// <summary>
        /// Gets or sets the section name.
        /// </summary>
        [JsonPropertyName("seccion")]
        public required string Section { get; set; }

        /// <summary>
        /// Gets or sets the section title.
        /// </summary>
        [JsonPropertyName("titulo")]
        public required string Title { get; set; }

        /// <summary>
        /// Gets or sets the section content.
        /// </summary>
        [JsonPropertyName("contenido")]
        public string? Conent { get; set; }

        /// <summary>
        /// Gets or sets the section order.
        /// </summary>
        [JsonPropertyName("orden")]
        public int Order { get; set; }
    }

}
