using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Meditox.Drug.Manager.Model
{
    public class Drug
    {
        public List<TechnicalDrugInfo> TechnicalDrugInfos { get; set; }

        public string nregistro { get; set; }
        public string nombre { get; set; }
        public string pactivos { get; set; }
        public string labtitular { get; set; }
        public string labcomercializador { get; set; }
        public string cpresc { get; set; }
        public Estado estado { get; set; }
        public bool comerc { get; set; }
        public bool receta { get; set; }
        public bool generico { get; set; }
        public bool conduc { get; set; }
        public bool triangulo { get; set; }
        public bool huerfano { get; set; }
        public bool biosimilar { get; set; }
        public Nosustituible nosustituible { get; set; }
        public bool psum { get; set; }
        public bool notas { get; set; }
        public bool materialesInf { get; set; }
        public bool ema { get; set; }
        public object[] docs { get; set; }
        public Atc[] atcs { get; set; }
        public Principiosactivo[] principiosActivos { get; set; }
        public Excipiente[] excipientes { get; set; }
        public Viasadministracion[] viasAdministracion { get; set; }
        public Presentacione[] presentaciones { get; set; }
        public Formafarmaceutica formaFarmaceutica { get; set; }
        public Formafarmaceuticasimplificada formaFarmaceuticaSimplificada { get; set; }
        public Vtm vtm { get; set; }
        public string dosis { get; set; }
    }

    public class Estado
    {
        public long aut { get; set; }
    }

    public class Nosustituible
    {
        public int id { get; set; }
        public string nombre { get; set; }
    }

    public class Formafarmaceutica
    {
        public int id { get; set; }
        public string nombre { get; set; }
    }

    public class Formafarmaceuticasimplificada
    {
        public int id { get; set; }
        public string nombre { get; set; }
    }

    public class Vtm
    {
        public long id { get; set; }
        public string nombre { get; set; }
    }

    public class Atc
    {
        public string codigo { get; set; }
        public string nombre { get; set; }
        public int nivel { get; set; }
    }

    public class Principiosactivo
    {
        public int id { get; set; }
        public string codigo { get; set; }
        public string nombre { get; set; }
        public string cantidad { get; set; }
        public string unidad { get; set; }
        public int orden { get; set; }
    }

    public class Excipiente
    {
        public int id { get; set; }
        public string nombre { get; set; }
        public string cantidad { get; set; }
        public string unidad { get; set; }
        public int orden { get; set; }
    }

    public class Viasadministracion
    {
        public int id { get; set; }
        public string nombre { get; set; }
    }

    public class Presentacione
    {
        public string cn { get; set; }
        public string nombre { get; set; }
        public Estado1 estado { get; set; }
        public bool comerc { get; set; }
        public bool psum { get; set; }
    }

    public class Estado1
    {
        public long aut { get; set; }
    }

}
