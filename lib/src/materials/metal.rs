#![allow(clippy::excessive_precision)]
use crate::Spectrum;

#[derive(Default, Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum Metal {
    #[default]
    SILVER,
    ALUMINIUM,
    GOLD,
    COPPER,
    IRON,
    MERCURY,
    LEAD,
    PLATINUM,
    TUNGSTEN,
    BERYLLIUM,
    BISMUTH,
    COBALT,
    CHROMIUM,
    GERMANIUM,
    POTASSIUM,
    LITHIUM,
    MAGNESIUM,
    MANGANESE,
    MOLYBDENUM,
    SODIUM,
    NIOBIUM,
    NICKEL,
    PALLADIUM,
    RHODIUM,
    TANTALUM,
    TITANIUM,
    VANADIUM,
    ZINC,
    ZIRCONIUM,
}

impl Metal {
    pub fn all_types() -> [Self; 29] {
        [
            Metal::SILVER,
            Metal::ALUMINIUM,
            Metal::GOLD,
            Metal::COPPER,
            Metal::IRON,
            Metal::MERCURY,
            Metal::LEAD,
            Metal::PLATINUM,
            Metal::TUNGSTEN,
            Metal::BERYLLIUM,
            Metal::BISMUTH,
            Metal::COBALT,
            Metal::CHROMIUM,
            Metal::GERMANIUM,
            Metal::POTASSIUM,
            Metal::LITHIUM,
            Metal::MAGNESIUM,
            Metal::MANGANESE,
            Metal::MOLYBDENUM,
            Metal::SODIUM,
            Metal::NIOBIUM,
            Metal::NICKEL,
            Metal::PALLADIUM,
            Metal::RHODIUM,
            Metal::TANTALUM,
            Metal::TITANIUM,
            Metal::VANADIUM,
            Metal::ZINC,
            Metal::ZIRCONIUM,
        ]
    }

    pub fn name(&self) -> &str {
        match self {
            Metal::SILVER => "Silver",
            Metal::ALUMINIUM => "Aluminium",
            Metal::GOLD => "Gold",
            Metal::COPPER => "Copper",
            Metal::IRON => "Iron",
            Metal::MERCURY => "Mercury",
            Metal::LEAD => "Lead",
            Metal::PLATINUM => "Platinum",
            Metal::TUNGSTEN => "Tungsten",
            Metal::BERYLLIUM => "Beryllium",
            Metal::BISMUTH => "Bismuth",
            Metal::COBALT => "Cobalt",
            Metal::CHROMIUM => "Chromium",
            Metal::GERMANIUM => "Germanium",
            Metal::POTASSIUM => "Potassium",
            Metal::LITHIUM => "Lithium",
            Metal::MAGNESIUM => "Magnesium",
            Metal::MANGANESE => "Manganese",
            Metal::MOLYBDENUM => "Moybdenum",
            Metal::SODIUM => "Sodium",
            Metal::NIOBIUM => "Niobium",
            Metal::NICKEL => "Nickel",
            Metal::PALLADIUM => "Palladium",
            Metal::RHODIUM => "Rhodium",
            Metal::TANTALUM => "Tantalum",
            Metal::TITANIUM => "Titanium",
            Metal::VANADIUM => "Vanadium",
            Metal::ZINC => "Zinc",
            Metal::ZIRCONIUM => "Zirconium",
        }
    }

    pub fn index_of_refraction(&self) -> Spectrum {
        match self {
            Metal::SILVER => Spectrum::from([
                0.0493475, 0.0416025, 0.0410099, 0.0484151, 0.0500000, 0.0500000, 0.0532925,
                0.0583626, 0.0536119, 0.0523413, 0.0579066, 0.0569087, 0.0522249, 0.0476667,
                0.0432222, 0.0389462,
            ]),
            Metal::ALUMINIUM => Spectrum::from([
                0.5135442, 0.5703716, 0.6333536, 0.7036586, 0.7759180, 0.8493804, 0.9302864,
                1.0151918, 1.1065556, 1.2102959, 1.3143414, 1.4303082, 1.5590260, 1.6975143,
                1.8419621, 2.0029783,
            ]),
            Metal::GOLD => Spectrum::from([
                1.4621289, 1.4438605, 1.3831228, 1.3007174, 1.1025913, 0.8031203, 0.5577828,
                0.4291050, 0.3405671, 0.2719932, 0.2258478, 0.1883606, 0.1555826, 0.1376667,
                0.1332222, 0.1312788,
            ]),
            Metal::COPPER => Spectrum::from([
                1.2891678, 1.2537014, 1.2424659, 1.2461510, 1.2270668, 1.1974400, 1.1273525,
                0.9996451, 0.8155821, 0.6087157, 0.3808574, 0.2752693, 0.2378089, 0.2176667,
                0.2132222, 0.2136113,
            ]),
            Metal::IRON => Spectrum::from([
                2.3298004, 2.4692903, 2.5814624, 2.6635699, 2.7239997, 2.8072000, 2.8888891,
                2.9425745, 2.9436364, 2.9261448, 2.8926003, 2.8923810, 2.9113717, 2.9053330,
                2.8786664, 2.8617642,
            ]),
            Metal::MERCURY => Spectrum::from([
                0.8859045, 0.9710537, 1.0606544, 1.1540451, 1.2512437, 1.3505762, 1.4511074,
                1.5522829, 1.6539246, 1.7564576, 1.8590126, 1.9646540, 2.0733328, 2.1820116,
                2.2904751, 2.3979113,
            ]),
            Metal::LEAD => Spectrum::from([
                0.3249431, 0.3554998, 0.3885345, 0.4294022, 0.4722900, 0.5305417, 0.5961630,
                0.6662911, 0.7699863, 0.8805264, 0.9910657, 1.0560176, 1.0760944, 1.0961711,
                1.1162480, 1.1337140,
            ]),
            Metal::PLATINUM => Spectrum::from([
                0.9186384, 0.7606118, 0.6407908, 0.5785009, 0.5282220, 0.5001245, 0.4826666,
                0.4668670, 0.4634193, 0.4624901, 0.4615610, 0.4656752, 0.4747538, 0.4838323,
                0.4929109, 0.5031975,
            ]),
            Metal::TUNGSTEN => Spectrum::from([
                1.6186066, 1.9181926, 2.1916623, 2.1743488, 2.0957577, 1.8867785, 1.6152557,
                1.3594210, 1.2204373, 1.1052804, 0.9901245, 0.9288915, 0.9207388, 0.9125861,
                0.9044334, 0.9060270,
            ]),
            Metal::BERYLLIUM => Spectrum::from([
                2.9243031, 3.0058403, 3.0766749, 3.1380355, 3.1913092, 3.2372100, 3.2768426,
                3.3107553, 3.3395073, 3.3637276, 3.3838532, 3.4001961, 3.4131138, 3.4229360,
                3.4298885, 3.4342523,
            ]),
            Metal::BISMUTH => Spectrum::from([
                0.5484297, 0.4668909, 0.3973810, 0.3527036, 0.3133399, 0.2877015, 0.2686467,
                0.2508102, 0.2420491, 0.2351386, 0.2282281, 0.2249593, 0.2252755, 0.2255917,
                0.2259078, 0.2270003,
            ]),
            Metal::COBALT => Spectrum::from([
                1.0047715, 0.9257409, 0.8492233, 0.7854902, 0.7247291, 0.6786140, 0.6395243,
                0.6025591, 0.5814173, 0.5635021, 0.5455870, 0.5374967, 0.5390775, 0.5406584,
                0.5422392, 0.5474628,
            ]),
            Metal::CHROMIUM => Spectrum::from([
                2.0663610, 2.1856887, 2.3270500, 2.5022149, 2.6916001, 2.8564003, 3.0171671,
                3.1663094, 3.2054543, 3.2083073, 3.1798930, 3.1452382, 3.1071682, 3.0802221,
                3.0624442, 3.0540481,
            ]),
            Metal::GERMANIUM => Spectrum::from([
                4.1736498, 4.1046996, 4.1240249, 4.2116251, 4.3626251, 4.5970001, 4.9554992,
                5.2225003, 5.4393754, 5.7634997, 5.7071252, 5.5001249, 5.3199997, 5.1741247,
                5.0604997, 4.9660006,
            ]),
            Metal::POTASSIUM => Spectrum::from([
                0.0410330, 0.0419204, 0.0426030, 0.0427998, 0.0443553, 0.0461193, 0.0478399,
                0.0494908, 0.0509475, 0.0523921, 0.0525978, 0.0516067, 0.0506155, 0.0494914,
                0.0481754, 0.0468578,
            ]),
            Metal::LITHIUM => Spectrum::from([
                0.2655967, 0.2532133, 0.2401011, 0.2288836, 0.2176100, 0.2106487, 0.2048686,
                0.2006800, 0.1978172, 0.1942308, 0.1920000, 0.1906970, 0.1901620, 0.1904722,
                0.1911706, 0.1925750,
            ]),
            Metal::MAGNESIUM => Spectrum::from([
                0.1828494, 0.2000048, 0.2191723, 0.2383398, 0.2575073, 0.2766747, 0.2958422,
                0.3150097, 0.3341772, 0.3533446, 0.3725124, 0.4019851, 0.4415593, 0.4811336,
                0.5207078, 0.5602821,
            ]),
            Metal::MANGANESE => Spectrum::from([
                1.9935207, 2.0549791, 2.1105373, 2.1827598, 2.2357602, 2.2892001, 2.3424926,
                2.3922710, 2.4409091, 2.4790146, 2.5020108, 2.5254760, 2.5492892, 2.5746665,
                2.6013331, 2.6293128,
            ]),
            Metal::MOLYBDENUM => Spectrum::from([
                0.4255734, 0.4363360, 0.4506717, 0.4731192, 0.4974073, 0.5295401, 0.5654359,
                0.6028720, 0.6517810, 0.7030295, 0.7542783, 0.8173596, 0.8920885, 0.9668176,
                1.0415466, 1.1235683,
            ]),
            Metal::SODIUM => Spectrum::from([
                0.0806395, 0.0808920, 0.0770582, 0.0710897, 0.0655496, 0.0604744, 0.0558647,
                0.0519640, 0.0488073, 0.0465147, 0.0449333, 0.0436545, 0.0426100, 0.0421167,
                0.0418338, 0.0418000,
            ]),
            Metal::NIOBIUM => Spectrum::from([
                1.5550001, 1.7600000, 1.9549999, 2.0100000, 2.0750000, 2.1650000, 2.2150002,
                2.2600000, 2.2900000, 2.2800000, 2.2750001, 2.2750001, 2.2700000, 2.2500000,
                2.2249999, 2.1950002,
            ]),
            Metal::NICKEL => Spectrum::from([
                1.3444345, 1.2322421, 1.1401201, 1.0946163, 1.0592464, 1.0547223, 1.0649939,
                1.0790211, 1.1210229, 1.1687291, 1.2164357, 1.2844858, 1.3725616, 1.4606376,
                1.5487134, 1.6472706,
            ]),
            Metal::PALLADIUM => Spectrum::from([
                1.0338617, 0.9183672, 0.8037344, 0.7079729, 0.6166075, 0.5471541, 0.4882111,
                0.4318375, 0.3946017, 0.3612683, 0.3279351, 0.3028529, 0.2858926, 0.2689324,
                0.2519721, 0.2366281,
            ]),
            Metal::RHODIUM => Spectrum::from([
                1.4980031, 1.6410450, 1.7506218, 1.8302091, 1.8704268, 1.8938577, 1.9238360,
                1.9666941, 2.0117514, 2.0504222, 2.0965085, 2.1447096, 2.1940718, 2.2480664,
                2.3039436, 2.3628149,
            ]),
            Metal::TANTALUM => Spectrum::from([
                1.3093551, 1.2771892, 1.2457765, 1.2173707, 1.1896722, 1.1656814, 1.1434687,
                1.1218592, 1.1047412, 1.0885391, 1.0723371, 1.0589964, 1.0484724, 1.0379486,
                1.0274246, 1.0177078,
            ]),
            Metal::TITANIUM => Spectrum::from([
                0.8133286, 0.6581362, 0.5291589, 0.4527608, 0.3874686, 0.3465812, 0.3174001,
                0.2901124, 0.2769287, 0.2666210, 0.2563133, 0.2518032, 0.2530001, 0.2541971,
                0.2553940, 0.2582493,
            ]),
            Metal::VANADIUM => Spectrum::from([
                1.0383094, 0.9769769, 0.9116813, 0.8450115, 0.7782991, 0.7188979, 0.6630036,
                0.6087407, 0.5666289, 0.5269947, 0.4873606, 0.4554314, 0.4310862, 0.4067411,
                0.3823961, 0.3599678,
            ]),
            Metal::ZINC => Spectrum::from([
                0.5979714, 0.6369423, 0.6783427, 0.7245890, 0.7719113, 0.8231625, 0.8762984,
                0.9300285, 0.9881851, 1.0472443, 1.1063034, 1.1685023, 1.2337913, 1.2990806,
                1.3643696, 1.4307468,
            ]),
            Metal::ZIRCONIUM => Spectrum::from([
                1.7850001, 1.8854999, 1.9844999, 2.0840001, 2.1894998, 2.3004999, 2.4130001,
                2.5314999, 2.6619999, 2.7940001, 2.9124999, 3.0230000, 3.1285000, 3.2265000,
                3.3189998, 3.4060001,
            ]),
        }
    }

    pub fn absorption(&self) -> Spectrum {
        match self {
            Metal::SILVER => Spectrum::from([
                2.2301846, 2.4536009, 2.6500175, 2.8523059, 3.0390980, 3.2232838, 3.4100761,
                3.5948911, 3.7597549, 3.9249725, 4.0942011, 4.2543230, 4.4093590, 4.5658331,
                4.7236109, 4.8811231,
            ]),
            Metal::ALUMINIUM => Spectrum::from([
                4.9599562, 5.2085543, 5.4542770, 5.6952329, 5.9313440, 6.1647310, 6.3964543,
                6.6272826, 6.8546181, 7.0752578, 7.2957330, 7.5080805, 7.7104082, 7.8930426,
                8.0657701, 8.2151413,
            ]),
            Metal::GOLD => Spectrum::from([
                1.9556787, 1.9458420, 1.9123862, 1.8580940, 1.8412964, 1.9728720, 2.2039385,
                2.4696045, 2.7156329, 2.9560721, 3.1913807, 3.4033818, 3.6024392, 3.7917333,
                3.9721780, 4.1496315,
            ]),
            Metal::COPPER => Spectrum::from([
                2.1880841, 2.2994711, 2.3922899, 2.4754829, 2.5438442, 2.5888162, 2.5977578,
                2.5913863, 2.6581283, 2.8192344, 3.1063750, 3.3725505, 3.6263988, 3.8538666,
                4.0574222, 4.2534952,
            ]),
            Metal::IRON => Spectrum::from([
                2.6355989, 2.7059917, 2.7659874, 2.8172975, 2.8651199, 2.8968000, 2.9164166,
                2.9340539, 2.9681816, 3.0037038, 3.0378821, 3.0654762, 3.0893059, 3.1219997,
                3.1619999, 3.2002769,
            ]),
            Metal::MERCURY => Spectrum::from([
                3.5062423, 3.6874237, 3.8626888, 4.0317068, 4.1943846, 4.3533735, 4.5043602,
                4.6510468, 4.7960143, 4.9376955, 5.0792956, 5.2128019, 5.3383427, 5.4638824,
                5.5869341, 5.6981401,
            ]),
            Metal::LEAD => Spectrum::from([
                4.0195465, 4.2509179, 4.4820976, 4.7135978, 4.9451337, 5.1758041, 5.4060593,
                5.6317062, 5.8230267, 6.0073471, 6.1916671, 6.3398685, 6.4525146, 6.5651617,
                6.6778092, 6.8047838,
            ]),
            Metal::PLATINUM => Spectrum::from([
                3.1870580, 3.4565804, 3.7471020, 4.0306149, 4.3123531, 4.5813990, 4.8443556,
                5.1058297, 5.3562531, 5.6044235, 5.8525944, 6.0953822, 6.3328705, 6.5703592,
                6.8078485, 7.0440154,
            ]),
            Metal::TUNGSTEN => Spectrum::from([
                5.0224886, 5.1223469, 5.1266627, 4.9920387, 4.8368349, 4.8983498, 5.0638180,
                5.2457008, 5.5498366, 5.8789015, 6.2079668, 6.5348244, 6.8595076, 7.1841917,
                7.5088758, 7.8302183,
            ]),
            Metal::BERYLLIUM => Spectrum::from([
                3.1354477, 3.1305034, 3.1265838, 3.1242135, 3.1236978, 3.1252010, 3.1289186,
                3.1348073, 3.1429076, 3.1532235, 3.1657586, 3.1804538, 3.1971717, 3.2160277,
                3.2368176, 3.2596068,
            ]),
            Metal::BISMUTH => Spectrum::from([
                2.6752372, 2.8655274, 3.0637200, 3.2657769, 3.4684854, 3.6677101, 3.8652637,
                4.0620871, 4.2534695, 4.4437418, 4.6340151, 4.8210640, 5.0049391, 5.1888146,
                5.3726902, 5.5557137,
            ]),
            Metal::COBALT => Spectrum::from([
                3.2935798, 3.4702916, 3.6574616, 3.8640819, 4.0748148, 4.2946935, 4.5189600,
                4.7437081, 4.9720421, 5.2011070, 5.4301724, 5.6592484, 5.8883362, 6.1174235,
                6.3465109, 6.5751114,
            ]),
            Metal::CHROMIUM => Spectrum::from([
                2.9110799, 3.0316389, 3.1350002, 3.2297351, 3.2851200, 3.3167999, 3.3299699,
                3.3272500, 3.3109088, 3.3000908, 3.3002143, 3.3123810, 3.3314323, 3.3522220,
                3.3744445, 3.3932481,
            ]),
            Metal::GERMANIUM => Spectrum::from([
                2.1280499, 2.1184497, 2.1777248, 2.2538750, 2.3291249, 2.3937500, 2.3559999,
                2.1146250, 1.9512501, 1.5452501, 1.0250000, 0.7417501, 0.5866250, 0.4902500,
                0.4248750, 0.3763750,
            ]),
            Metal::POTASSIUM => Spectrum::from([
                0.8335377, 0.9592336, 1.0602843, 1.1394393, 1.2197677, 1.2985948, 1.3732564,
                1.4464669, 1.5160604, 1.5854282, 1.6554775, 1.7261851, 1.7968925, 1.8715723,
                1.9519920, 2.0324588,
            ]),
            Metal::LITHIUM => Spectrum::from([
                1.8254000, 1.9152000, 2.0293026, 2.1475050, 2.2847469, 2.3971295, 2.5185759,
                2.6283998, 2.7300861, 2.8586535, 2.9800000, 3.0945454, 3.2037601, 3.3144448,
                3.4238751, 3.5250001,
            ]),
            Metal::MAGNESIUM => Spectrum::from([
                3.6272044, 3.8281319, 4.0294867, 4.2308426, 4.4321976, 4.6335526, 4.8349080,
                5.0362635, 5.2376184, 5.4389744, 5.6403298, 5.8484764, 6.0632801, 6.2780848,
                6.4928885, 6.7076921,
            ]),
            Metal::MANGANESE => Spectrum::from([
                2.8043194, 2.8946557, 2.9760001, 3.0543799, 3.1209598, 3.1904001, 3.2621422,
                3.3319345, 3.3936362, 3.4551544, 3.5178821, 3.5740476, 3.6264355, 3.6817780,
                3.7395558, 3.7948449,
            ]),
            Metal::MOLYBDENUM => Spectrum::from([
                5.2792678, 5.5915356, 5.9038429, 6.2164774, 6.5292025, 6.8427553, 7.1567068,
                7.4709573, 7.7874312, 8.1043587, 8.4212885, 8.7406902, 9.0625286, 9.3843670,
                9.7062054, 10.0286808,
            ]),
            Metal::SODIUM => Spectrum::from([
                1.5470333, 1.6482666, 1.7495611, 1.8491914, 1.9550426, 2.0634844, 2.1768005,
                2.2872000, 2.3874583, 2.4801280, 2.5733333, 2.6684847, 2.7656400, 2.8650000,
                2.9642437, 3.0597501,
            ]),
            Metal::NIOBIUM => Spectrum::from([
                2.9700000, 2.9650002, 2.9900000, 3.0200000, 3.0549998, 3.0749998, 3.1150000,
                3.1600001, 3.2000000, 3.2400002, 3.2500000, 3.2750001, 3.3299999, 3.3699996,
                3.3950000, 3.4250000,
            ]),
            Metal::NICKEL => Spectrum::from([
                3.2570171, 3.4973893, 3.7570717, 4.0355148, 4.3177423, 4.6029634, 4.8896208,
                5.1764579, 5.4646297, 5.7530737, 6.0415177, 6.3329391, 6.6272936, 6.9216475,
                7.2160020, 7.5120621,
            ]),
            Metal::PALLADIUM => Spectrum::from([
                2.9739745, 3.1187980, 3.2821949, 3.4789016, 3.6825187, 3.8975646, 4.1180935,
                4.3386765, 4.5596728, 4.7807527, 5.0018330, 5.2200060, 5.4353175, 5.6506290,
                5.8659410, 6.0801287,
            ]),
            Metal::RHODIUM => Spectrum::from([
                4.2656069, 4.3690867, 4.4522600, 4.5270605, 4.6193748, 4.7387910, 4.8752851,
                5.0156326, 5.1554227, 5.2972322, 5.4395261, 5.5872178, 5.7394891, 5.8849726,
                6.0300989, 6.1822720,
            ]),
            Metal::TANTALUM => Spectrum::from([
                3.6245697, 3.7929885, 3.9649055, 4.1438866, 4.3243952, 4.5093346, 4.6963987,
                4.8838816, 5.0744772, 5.2657080, 5.4569397, 5.6491642, 5.8423662, 6.0355692,
                6.2287712, 6.4220476,
            ]),
            Metal::TITANIUM => Spectrum::from([
                2.5880985, 2.8023322, 3.0375340, 3.2785912, 3.5205312, 3.7538388, 3.9830050,
                4.2108741, 4.4290762, 4.6453075, 4.8615384, 5.0729475, 5.2796092, 5.4862714,
                5.6929340, 5.8984232,
            ]),
            Metal::VANADIUM => Spectrum::from([
                3.3230336, 3.4604707, 3.6065071, 3.7740052, 3.9461894, 4.1332622, 4.3274765,
                4.5229006, 4.7273350, 4.9336071, 5.1398783, 5.3478608, 5.5575266, 5.7671919,
                5.9768581, 6.1862721,
            ]),
            Metal::ZINC => Spectrum::from([
                3.6020923, 3.8081672, 4.0114069, 4.2089186, 4.4051561, 4.5966659, 4.7859077,
                4.9744172, 5.1574697, 5.3394089, 5.5213485, 5.6991763, 5.8729572, 6.0467377,
                6.2205181, 6.3926554,
            ]),
            Metal::ZIRCONIUM => Spectrum::from([
                2.6434999, 2.7624998, 2.8715000, 2.9795001, 3.0855000, 3.1830001, 3.2735000,
                3.3600001, 3.4334998, 3.4845002, 3.5200000, 3.5514998, 3.5759997, 3.5945001,
                3.6095002, 3.6194999,
            ]),
        }
    }
}

impl From<u8> for Metal {
    fn from(id: u8) -> Self {
        match id {
            0 => Metal::SILVER,
            1 => Metal::ALUMINIUM,
            2 => Metal::GOLD,
            3 => Metal::COPPER,
            4 => Metal::IRON,
            5 => Metal::MERCURY,
            6 => Metal::LEAD,
            7 => Metal::PLATINUM,
            8 => Metal::TUNGSTEN,
            9 => Metal::BERYLLIUM,
            10 => Metal::BISMUTH,
            11 => Metal::COBALT,
            12 => Metal::CHROMIUM,
            13 => Metal::GERMANIUM,
            14 => Metal::POTASSIUM,
            15 => Metal::LITHIUM,
            16 => Metal::MAGNESIUM,
            17 => Metal::MANGANESE,
            18 => Metal::MOLYBDENUM,
            19 => Metal::SODIUM,
            20 => Metal::NIOBIUM,
            21 => Metal::NICKEL,
            22 => Metal::PALLADIUM,
            23 => Metal::RHODIUM,
            24 => Metal::TANTALUM,
            25 => Metal::TITANIUM,
            26 => Metal::VANADIUM,
            27 => Metal::ZINC,
            28 => Metal::ZIRCONIUM,
            _ => Metal::default(),
        }
    }
}

impl From<Metal> for u8 {
    fn from(metal: Metal) -> Self {
        match metal {
            Metal::SILVER => 0,
            Metal::ALUMINIUM => 1,
            Metal::GOLD => 2,
            Metal::COPPER => 3,
            Metal::IRON => 4,
            Metal::MERCURY => 5,
            Metal::LEAD => 6,
            Metal::PLATINUM => 7,
            Metal::TUNGSTEN => 8,
            Metal::BERYLLIUM => 9,
            Metal::BISMUTH => 10,
            Metal::COBALT => 11,
            Metal::CHROMIUM => 12,
            Metal::GERMANIUM => 13,
            Metal::POTASSIUM => 14,
            Metal::LITHIUM => 15,
            Metal::MAGNESIUM => 16,
            Metal::MANGANESE => 17,
            Metal::MOLYBDENUM => 18,
            Metal::SODIUM => 19,
            Metal::NIOBIUM => 20,
            Metal::NICKEL => 21,
            Metal::PALLADIUM => 22,
            Metal::RHODIUM => 23,
            Metal::TANTALUM => 24,
            Metal::TITANIUM => 25,
            Metal::VANADIUM => 26,
            Metal::ZINC => 27,
            Metal::ZIRCONIUM => 28,
        }
    }
}
