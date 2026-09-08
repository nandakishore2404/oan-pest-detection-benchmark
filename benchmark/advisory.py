# -*- coding: utf-8 -*-
"""
Kenya Agronomic Advisory Knowledge Base & Decision Engine
=========================================================
Generates actionable, evidence-based agronomic advisories based on computer-vision
diagnoses for smallholder farming systems in Kenya.
Formulated in alignment with KALRO, PCPB Kenya, CABI Plantwise, and CGIAR IPM standards.
"""

from typing import Any, Dict, List, Optional

# Master Agronomic Knowledge Base for Kenya Priority Crops
KENYA_AGRI_ADVISORY_DB = {
    "fall armyworm": {
        "common_name": "Fall Armyworm",
        "scientific_name": "Spodoptera frugiperda",
        "primary_crop": "Maize (Zea mays), Sorghum, Millets",
        "swahili_name": "Kiwavi Jeshi Vamizi",
        "symptoms": (
            "Ragged 'windowpane' feeding on young leaves, extensive whorl damage with moist sawdust-like frass "
            "(caterpillar excrement), and feeding on developing tassels and ear cobs."
        ),
        "damage_risk": "High - Can cause 30% to 70% yield loss in maize if uncontrolled during vegetative stages V4-V8.",
        "cultural_actions": [
            "Hand-pick and crush egg masses (covered in greyish scales) and visible caterpillars in early morning or evening.",
            "Apply a handful of clean dry sand, wood ash, or soil into the whorls of infested plants to suffocate larvae.",
            "Maintain clean fields: rogue volunteer maize plants and eliminate alternative grassy weed hosts (Digitaria, Eleusine).",
            "Plant early with the first seasonal rains to escape peak moth oviposition flights."
        ],
        "biological_controls": [
            "Apply biopesticides containing Bacillus thuringiensis (Bt) kurstaki or aizawai when larvae are small (1st-2nd instar).",
            "Apply Beauveria bassiana or Metarhizium anisopliae fungal entomopathogen formulations in humid conditions.",
            "Apply 5% Neem Seed Kernel Extract (NSKE) or commercial Azadirachtin formulations into the funnel whorl.",
            "Conserve natural predators and parasitoids: parasitic wasps (Telenomus remus, Trichogramma chilonis), earwigs, and predatory bugs."
        ],
        "chemical_interventions": [
            {
                "active_ingredient": "Chlorantraniliprole 200 g/L (Coragen 20 SC or equivalent)",
                "application_timing": "Apply at early whorl infestation when threshold reaches 10-20% damaged plants.",
                "phi_days": "14 days",
                "safety_note": "Rotate with different MoA groups (Group 28) to prevent insecticide resistance."
            },
            {
                "active_ingredient": "Emamectin Benzoate 50 g/kg (Group 6)",
                "application_timing": "Direct coarse spray targeted straight into the leaf whorl funnels where larvae hide.",
                "phi_days": "7 days",
                "safety_note": "Wear complete PPE (gloves, face shield, gumboots). Do not spray during wind or midday heat."
            },
            {
                "active_ingredient": "Flubendiamide 480 g/L (Group 28)",
                "application_timing": "Target early instar larvae (under 1 cm length). Ineffective against deeply bored larvae.",
                "phi_days": "14 days",
                "safety_note": "Ensure water pH is near neutral (6.5-7.0) for optimal insecticide stability."
            }
        ],
        "preventative_practices": [
            "Adopt 'Push-Pull Technology' developed by icipe: intercrop maize with Desmodium (silverleaf/greenleaf) to repel moths and plant Napier or Brachiaria grass borders to trap them.",
            "Intercrop maize with common beans, cowpeas, or pigeon peas to disrupt pest host-seeking behavior.",
            "Practice field scouting twice weekly by inspecting 20 plants in 5 separate field locations in a 'W' pattern."
        ],
        "extension_escalation": (
            "If whorl damage exceeds 25% across your field, notify your local Ward Agricultural Officer or contact the "
            "KALRO Advisory Helpline (Toll-Free 0800 721 741) for regional outbreak mapping."
        )
    },
    "stalk borer": {
        "common_name": "African Maize Stalk Borer",
        "scientific_name": "Busseola fusca",
        "primary_crop": "Maize, Sorghum",
        "swahili_name": "Funza wa Mabua",
        "symptoms": (
            "Pinholes in straight rows across unfolding leaves, 'dead heart' (wilting and drying of the central growing shoot), "
            "and exit holes along the stem with sawdust-like frass."
        ),
        "damage_risk": "Moderate to High - Stem tunneling causes lodging (falling over) and sterile or stunted cobs.",
        "cultural_actions": [
            "Cut down and destroy or ensile maize stalks and stubble immediately after harvest to kill diapausing larvae.",
            "Pull out and burn plants showing 'dead heart' symptoms to destroy larvae before they tunnel down into the stem base.",
            "Rotate cereal crops with legumes or sunflowers for at least one full growing season."
        ],
        "biological_controls": [
            "Conserve native larval parasitoids: Cotesia sesamiae (wasps that parasitize up to 50% of borer larvae).",
            "Apply Neem seed powder or bio-insecticide into the plant whorls before larvae enter the stalk."
        ],
        "chemical_interventions": [
            {
                "active_ingredient": "Beta-cyfluthrin or Deltamethrin granules",
                "application_timing": "Apply into the whorls 2 to 3 weeks after crop emergence before stem penetration.",
                "phi_days": "14 days",
                "safety_note": "Once the larva penetrates the stem, chemical sprays are completely ineffective."
            }
        ],
        "preventative_practices": [
            "Implement Push-Pull intercropping (Desmodium + Napier grass).",
            "Use certified borer-tolerant hybrid maize varieties recommended for your agro-ecological zone."
        ],
        "extension_escalation": "Consult local KALRO center or sub-county crop officer for resistant seed varieties."
    },
    "angular leaf spot": {
        "common_name": "Bean Angular Leaf Spot",
        "scientific_name": "Pseudocercospora griseola",
        "primary_crop": "Common Bean (Phaseolus vulgaris)",
        "swahili_name": "Ugonjwa wa Madoa ya Pembe ya Maharagwe",
        "symptoms": (
            "Small angular, vein-delimited brown or necrotic spots on leaves. On pods, circular reddish-brown sunken lesions "
            "appear, leading to seed shriveling and staining."
        ),
        "damage_risk": "High - Can cause 40% to 80% bean yield loss during warm, wet, and humid seasons in Western and Rift Valley Kenya.",
        "cultural_actions": [
            "Always use certified disease-free bean seed (e.g. from Kenya Seed Company or Simlaw Seeds). Avoid saving seed from infected fields.",
            "Practice strict 2-year crop rotation with non-legumes (maize, sorghum, cassava, sweet potatoes).",
            "Do not work in or weed wet bean fields to prevent mechanical transmission of fungal conidia.",
            "Deep plow or burn infected crop debris after harvest to reduce soil inoculum survival."
        ],
        "biological_controls": [
            "Apply protective Copper Hydroxide or Copper Oxychloride sprays at first sign of vegetative spotting.",
            "Treat planting seeds with Trichoderma harzianum bio-fungicide inoculant before sowing."
        ],
        "chemical_interventions": [
            {
                "active_ingredient": "Azoxystrobin 200 g/L + Difenoconazole 125 g/L (Amistar Top or equivalent)",
                "application_timing": "Spray at flowering initiation and repeat after 14 days if wet weather persists.",
                "phi_days": "14 days",
                "safety_note": "Ensure thorough canopy coverage including lower leaf surfaces."
            },
            {
                "active_ingredient": "Mancozeb 800 g/kg (Dithane M-45)",
                "application_timing": "Preventative contact spray applied during high humidity or prolonged morning dew.",
                "phi_days": "14 days",
                "safety_note": "Apply before rain showers to form a protective fungicide film."
            }
        ],
        "preventative_practices": [
            "Plant resistant or tolerant varieties released by KALRO (e.g. KK8, KK15, Nyota, Angaza).",
            "Maintain optimal plant spacing (45 cm x 15 cm) to promote air circulation and quick leaf drying."
        ],
        "extension_escalation": "Contact your local agricultural extension service for certified disease-tolerant bean seed stock."
    },
    "late blight": {
        "common_name": "Potato / Tomato Late Blight",
        "scientific_name": "Phytophthora infestans",
        "primary_crop": "Irish Potato (Solanum tuberosum), Tomato",
        "swahili_name": "Ugonjwa wa Ukungu wa Viazi",
        "symptoms": (
            "Irregular water-soaked pale-green to dark brown lesions that expand rapidly across leaves and stems. "
            "A delicate white downy fungal mildew appears on the underside of leaves under humid or misty conditions."
        ),
        "damage_risk": "Critical - Can completely destroy an entire potato or tomato crop in 7 to 10 days under cool, moist weather.",
        "cultural_actions": [
            "Plant certified clean potato seed tubers; never plant tubers showing reddish-brown dry rot beneath the skin.",
            "Hill up soil around potato plants (high ridging) to create a physical barrier protecting tubers from washing spores.",
            "De-haulm (cut and remove potato vines) 10-14 days before harvest to prevent tuber contamination at digging.",
            "Destroy cull piles and self-sown volunteer potato plants near the field."
        ],
        "biological_controls": [
            "Apply preventative copper-based bio-fungicides (Copper oxychloride) prior to rain spells.",
            "Use Bacillus subtilis-based biofungicides for early vegetative suppression."
        ],
        "chemical_interventions": [
            {
                "active_ingredient": "Metalaxyl-M 40 g/kg + Mancozeb 640 g/kg (Ridomil Gold MZ or equivalent)",
                "application_timing": "Curative and systemic spray applied at the very first sign of foliar lesions.",
                "phi_days": "14 days for potato, 7 days for tomato",
                "safety_note": "Maximum 2-3 applications per season to prevent pathogen resistance."
            },
            {
                "active_ingredient": "Cymoxanil + Mancozeb (Curzate M)",
                "application_timing": "Post-infection curative treatment within 24-48 hours after rain / fog exposure.",
                "phi_days": "7 days",
                "safety_note": "Alternate systemic products with pure contact protectants (Chlorothalonil or Mancozeb)."
            }
        ],
        "preventative_practices": [
            "Plant resistant varieties bred by KALRO Tigoni / CIP (e.g., Shangi, Unica, Asante, Kenya Mpya).",
            "Scout daily during misty, foggy, or rainy conditions in high-altitude zones (Nyandarua, Meru, Nakuru, Narok)."
        ],
        "extension_escalation": "Alert County Agriculture Department or KALRO Tigoni Potato Research Centre immediately upon outbreak."
    },
    "early blight": {
        "common_name": "Tomato / Potato Early Blight",
        "scientific_name": "Alternaria solani",
        "primary_crop": "Tomato (Solanum lycopersicum), Potato",
        "swahili_name": "Madoa ya Mapema ya Nyanya",
        "symptoms": (
            "Dark brown to black circular spots with concentric target-board rings on older leaves. Surrounding tissue yellowing "
            "(chlorosis) causes premature defoliation from the base upward."
        ),
        "damage_risk": "Moderate to High - Causes significant defoliation, sunburn on exposed fruit, and stem collar rot.",
        "cultural_actions": [
            "Prune lower leaves (up to 30 cm from ground level) once plants establish to avoid soil splash and improve airflow.",
            "Stake and tie tomato vines to keep foliage off moist soil.",
            "Use drip irrigation rather than overhead sprinklers to prevent wetting the foliar canopy."
        ],
        "biological_controls": [
            "Foliar sprays of Trichoderma viride or Bacillus amyloliquefaciens.",
            "Copper hydroxide preventative applications."
        ],
        "chemical_interventions": [
            {
                "active_ingredient": "Difenoconazole 250 g/L (Score 250 EC)",
                "application_timing": "Apply at early lesion development; repeat after 10-14 days.",
                "phi_days": "7 days for tomato",
                "safety_note": "Ensure safe harvest interval is strictly observed before marketing fruit."
            },
            {
                "active_ingredient": "Chlorothalonil 720 g/L",
                "application_timing": "Broad-spectrum contact protectant applied every 7-10 days during warm, humid periods.",
                "phi_days": "7 days",
                "safety_note": "Do not combine with foliar fertilizer oils."
            }
        ],
        "preventative_practices": [
            "Practice minimum 3-year rotation away from all Solanaceous crops (tomatoes, potatoes, peppers, eggplants).",
            "Mulch beds with clean dry straw or plastic mulch to create a splash barrier against soil-borne spores."
        ],
        "extension_escalation": "Consult local agro-dealer or agricultural extension officer for PCPB-registered fungicide rotation programs."
    },
    "slugs": {
        "common_name": "Slugs / Snails (Terrestrial Gastropods)",
        "scientific_name": "Gastropoda (Arionidae / Helicidae)",
        "primary_crop": "Vegetables, Brassicas, Maize Seedlings, Legumes",
        "swahili_name": "Konokono Asiye na Gamba / Konokono",
        "symptoms": (
            "Large irregular ragged holes chewed through leaf blades, rasping damage from radula, "
            "and silvery mucosal slime residue on foliage and surrounding soil."
        ),
        "damage_risk": "Moderate to High - Capable of defoliating seedlings and destroying commercial appearance of leafy crops.",
        "cultural_actions": [
            "Hand-collect slugs during early morning or damp dusk hours with a flashlight.",
            "Remove decomposing ground mulch, weeds, and damp wood debris touching crop stems.",
            "Apply rough abrasive barriers around vegetable beds (crushed eggshells, coarse sand, or fresh wood ash)."
        ],
        "biological_controls": [
            "Conserve natural predators including carabid ground beetles, frogs, toads, and domestic ducks.",
            "Apply entomopathogenic nematodes (Phasmarhabditis hermaphrodita) in damp soil beds."
        ],
        "chemical_interventions": [
            {
                "active_ingredient": "Iron (III) phosphate bait pellets 10 g/kg (Ferramol / Slugg-Off)",
                "application_timing": "Scatter bait evenly around plant base at dusk; safe for birds, hedgehogs, and domestic animals.",
                "phi_days": "0 days (Non-toxic mineral bait)",
                "safety_note": "Approved for organic and IPM certified production systems."
            },
            {
                "active_ingredient": "Metaldehyde 5% bait pellets",
                "application_timing": "Place pellets strictly inside covered bait traps/stations away from direct contact with edible foliage.",
                "phi_days": "7 days",
                "safety_note": "Toxic to dogs and domestic livestock; never scatter openly in livestock grazing areas."
            }
        ],
        "preventative_practices": [
            "Avoid evening sprinkler irrigation that creates damp overnight environments.",
            "Use drip irrigation to keep upper leaf canopy and bed surface dry."
        ],
        "extension_escalation": "Report severe localized gastropod outbreaks to Ward Extension Officers for perimeter bait trap distribution."
    },
    "caterpillars": {
        "common_name": "Caterpillars (Foliar Feeding Larvae)",
        "scientific_name": "Lepidoptera (Noctuidae / Pieridae)",
        "primary_crop": "Maize, Sorghum, Cabbages, Tomatoes, Beans",
        "swahili_name": "Viwavi wa Mboga na Nafaka",
        "symptoms": "Irregular foliar defoliation, windowpaning on young leaves, whorl feeding, and dark frass deposits.",
        "damage_risk": "High - Voracious foliar feeders capable of stripping vegetative leaf canopy in 48-72 hours.",
        "cultural_actions": [
            "Hand-pick and crush egg masses and young caterpillar clusters on leaf undersides.",
            "Intercrop maize with silverleaf desmodium (push-pull technology) or plant repellent borders."
        ],
        "biological_controls": [
            "Foliar application of Bacillus thuringiensis (Bt) kurstaki at 1-2 g/L.",
            "Apply neem seed kernel extract (Azadirachtin 0.03% EC)."
        ],
        "chemical_interventions": [
            {
                "active_ingredient": "Chlorantraniliprole 18.5% SC (Coragen)",
                "application_timing": "Apply when scouting confirms >10% plants with active young larvae.",
                "phi_days": "3 days",
                "safety_note": "Safe on beneficial predatory insects and parasitoid wasps."
            },
            {
                "active_ingredient": "Emamectin benzoate 5% SG",
                "application_timing": "Target early instar larvae inside whorls or under leaves.",
                "phi_days": "7 days",
                "safety_note": "Rotate chemical modes of action to prevent pesticide resistance."
            }
        ],
        "preventative_practices": [
            "Install pheromone traps to monitor adult moth arrival and peak flights."
        ],
        "extension_escalation": "Alert County Agriculture Department if widespread defoliation is observed across neighboring farms."
    },
    "beetles": {
        "common_name": "Beetles (Leaf Beetles & Weevils)",
        "scientific_name": "Coleoptera (Chrysomelidae / Curculionidae)",
        "primary_crop": "Cereals, Legumes, Solanaceae",
        "swahili_name": "Mende na Mbawakawa",
        "symptoms": "Shot-hole perforations, leaf skeletonization, flower feeding, and root/stem chewing.",
        "damage_risk": "Moderate - Causes foliar defoliation and vectors devastating viral pathogens.",
        "cultural_actions": [
            "Use yellow sticky traps and insect barrier netting over nursery beds.",
            "Rotate crops with non-host botanical families."
        ],
        "biological_controls": [
            "Apply entomopathogenic fungi (Beauveria bassiana).",
            "Spray botanical pyrethrum extracts or neem oil."
        ],
        "chemical_interventions": [
            {
                "active_ingredient": "Lambda-cyhalothrin 50 g/L (Karate 5 EC)",
                "application_timing": "Target adult feeding clusters during early morning hours.",
                "phi_days": "7 days",
                "safety_note": "Highly toxic to honeybees; never spray during active crop flowering."
            }
        ],
        "preventative_practices": [
            "Keep field perimeters free of wild solanaceous and brassica weed hosts."
        ],
        "extension_escalation": "Consult local extension officer for vector control recommendations."
    },
    "grasshoppers": {
        "common_name": "Grasshoppers & Locusts",
        "scientific_name": "Orthoptera (Acrididae)",
        "primary_crop": "Maize, Pasture, Sorghum, Millet, Vegetables",
        "swahili_name": "Nzige na Panzi",
        "symptoms": "Ragged leaf margin chewing, defoliation down to midribs, and severed young stems.",
        "damage_risk": "High - Rapid swarming consumption of green vegetative canopy.",
        "cultural_actions": [
            "Encourage free-range poultry/guinea fowl in vegetative plots to feed on nymphs.",
            "Till soil before planting to expose buried grasshopper egg pods to birds and sun."
        ],
        "biological_controls": [
            "Apply Metarhizium acridum (Novacrid / Green Muscle) bio-insecticide."
        ],
        "chemical_interventions": [
            {
                "active_ingredient": "Deltamethrin 25 g/L (Decis)",
                "application_timing": "Apply on nymph bands along field borders before swarms reach crop canopy.",
                "phi_days": "7 days",
                "safety_note": "Wear full protective PPE; avoid contaminating natural waterways."
            }
        ],
        "preventative_practices": [
            "Scout field borders during warm mornings when grasshoppers bask in sunlight."
        ],
        "extension_escalation": "Notify Sub-County Agricultural Officer immediately if gregarious locust swarms are sighted."
    },
    "weevils": {
        "common_name": "Weevils (Grain & Stem Weevils)",
        "scientific_name": "Curculionoidea (Sitophilus / Cylas sp.)",
        "primary_crop": "Maize, Sweet Potato, Stored Cereals",
        "swahili_name": "Mundu na Dumuzi",
        "symptoms": "Pinpoint exit holes in grains, hollowed stems, and granular frass powder.",
        "damage_risk": "High - Serious post-harvest grain destruction and sweet potato tuber tunneling.",
        "cultural_actions": [
            "Thoroughly dry harvested grain to below 13.5% moisture before storage.",
            "Store in certified hermetic grain bags (PICS bags) without chemical dust."
        ],
        "biological_controls": [
            "Treat tubers/soil with Beauveria bassiana."
        ],
        "chemical_interventions": [
            {
                "active_ingredient": "Pirimiphos-methyl 16 g/kg + Permethrin 3 g/kg (Actellic Gold Dust)",
                "application_timing": "Mix dust evenly with clean, dry stored grain (50 g per 90 kg bag).",
                "phi_days": "14 days",
                "safety_note": "Do not exceed dosage; ventilate storage areas thoroughly."
            }
        ],
        "preventative_practices": [
            "Deep plowing and high soil ridging around sweet potato tubers to prevent weevil entry."
        ],
        "extension_escalation": "Contact KALRO Post-Harvest Unit for hermetic storage demonstrations."
    },
    "earwigs": {
        "common_name": "Earwigs",
        "scientific_name": "Dermaptera (Forficulidae)",
        "primary_crop": "Soft fruits, Corn silks, Young seedlings",
        "swahili_name": "Kikope",
        "symptoms": "Chewed ragged edges on young leaves and silk clipping in maize cobs.",
        "damage_risk": "Low to Moderate - Often act as beneficial scavengers/aphid predators, but occasionally damage soft seedlings.",
        "cultural_actions": [
            "Set cardboard rolled traps or oil-filled saucers at base of crops to trap excess populations.",
            "Maintain balanced soil moisture."
        ],
        "biological_controls": [
            "Conserve natural bird and amphibian predators."
        ],
        "chemical_interventions": [
            {
                "active_ingredient": "Diatomaceous earth / Horticultural Neem oil",
                "application_timing": "Apply protective barrier around young vulnerable seedling stems.",
                "phi_days": "0 days",
                "safety_note": "Avoid synthetic chemicals; earwigs feed on aphids and mite pests."
            }
        ],
        "preventative_practices": [
            "Scout at night to confirm whether earwigs are feeding on crop tissue or predating other pests."
        ],
        "extension_escalation": "Escalate only if severe seedling destruction is verified."
    },
    "ants": {
        "common_name": "Ants (Formicidae)",
        "scientific_name": "Formicidae",
        "primary_crop": "Horticultural Crops, Fruit Trees, Vegetables",
        "swahili_name": "Sisimizi na Chungu",
        "symptoms": "Ant trails moving along stems, protection of aphid/scale colonies, and disturbed root zones.",
        "damage_risk": "Indirect - Guard sap-sucking aphids and mealybugs from beneficial ladybirds.",
        "cultural_actions": [
            "Control honeydew-producing aphids and mealybugs to naturally disperse ants.",
            "Apply sticky bands (tree tanglefoot) around woody stems."
        ],
        "biological_controls": [
            "Drench ant nests with boiling water or citrus peel extract drench."
        ],
        "chemical_interventions": [
            {
                "active_ingredient": "Boric acid + sugar bait liquid",
                "application_timing": "Place in covered bait stations along ant foraging trails.",
                "phi_days": "0 days",
                "safety_note": "Keep out of reach of children and livestock."
            }
        ],
        "preventative_practices": [
            "Inspect plants regularly for aphid clusters attended by ants."
        ],
        "extension_escalation": "Consult local extension officer for mealybug-ant symbiotic management."
    },
    "moths": {
        "common_name": "Adult Moths",
        "scientific_name": "Lepidoptera (Adults)",
        "primary_crop": "Field crops, Cereals, Vegetables",
        "swahili_name": "Nondo wa Mboga na Nafaka",
        "symptoms": "Adult moths fluttering near foliage or resting on leaves; egg deposition on leaf undersides.",
        "damage_risk": "Warning Sign - Adult moths do not chew foliage, but their eggs hatch into destructive caterpillars.",
        "cultural_actions": [
            "Install solar-powered light traps or kerosene water-pan traps at field perimeters.",
            "Inspect leaf undersides within 48 hours for fresh egg clusters and crush them."
        ],
        "biological_controls": [
            "Release Trichogramma egg parasitoid wasps to destroy moth eggs before hatch.",
            "Hang species-specific pheromone lure traps to disrupt mating."
        ],
        "chemical_interventions": [
            {
                "active_ingredient": "Neem oil extract 2% foliar spray",
                "application_timing": "Repels oviposition by female moths on upper leaves.",
                "phi_days": "0 days",
                "safety_note": "Do not spray broad-spectrum insecticides against adult moths; target larvae instead."
            }
        ],
        "preventative_practices": [
            "Maintain field border sanitation and avoid dense weed hosts."
        ],
        "extension_escalation": "Report heavy moth flights to Ward Agriculture Office for regional pest forecasting."
    },
    "bees": {
        "common_name": "Honey Bees (Beneficial Pollinator)",
        "scientific_name": "Apis mellifera",
        "primary_crop": "All Flowering Crops",
        "swahili_name": "Nyuki (Mdudu Mchavushaji Muhimu)",
        "symptoms": "Pollinator foraging on blossoms; zero feeding damage to foliage.",
        "damage_risk": "CRITICAL BENEFIT - Essential pollinator responsible for >35% crop yield; ZERO CROP DAMAGE.",
        "cultural_actions": [
            "PROTECT AND PRESERVE. Do NOT kill, disturb, or spray bees.",
            "Plant flowering pollinator borders (sunflowers, basil) to support hive health."
        ],
        "biological_controls": [
            "No control required. Maintain clean water sources for pollinators."
        ],
        "chemical_interventions": [],
        "preventative_practices": [
            "STRICT WARNING: Never spray insecticides (especially neonicotinoids or pyrethroids) during flowering or when bees are actively foraging."
        ],
        "extension_escalation": "If a wild swarm poses a direct hazard to farm workers, contact a registered local beekeeper or apiculture officer for safe relocation."
    },
    "wasps": {
        "common_name": "Parasitoid & Predatory Wasps (Beneficial)",
        "scientific_name": "Hymenoptera (Ichneumonidae / Braconidae / Vespidae)",
        "primary_crop": "All Agricultural Crops",
        "swahili_name": "Nyigu (Mwindaji Asilia wa Viwavi)",
        "symptoms": "Wasps hunting on foliage; parasitize caterpillar larvae and pupae; zero foliage damage.",
        "damage_risk": "CRITICAL BENEFIT - Natural biological control agent that suppresses pest populations naturally.",
        "cultural_actions": [
            "Conserve flowering nectar plants (dill, fennel, coriander) that provide energy for adult parasitoid wasps.",
            "Do not destroy harmless solitary wasp nests in field borders."
        ],
        "biological_controls": [
            "Protect native wasp populations as first-line defense against caterpillars and aphids."
        ],
        "chemical_interventions": [],
        "preventative_practices": [
            "Avoid broad-spectrum chemical sprays that wipe out parasitoid wasps and trigger secondary pest resurgence."
        ],
        "extension_escalation": "Consult KALRO Biological Control Unit for parasitoid augmentation programs."
    },
    "earthworms": {
        "common_name": "Earthworms (Beneficial Soil Organism)",
        "scientific_name": "Lumbricina",
        "primary_crop": "Agricultural Soils",
        "swahili_name": "Mnyoo wa Udongo (Mbolea Asilia)",
        "symptoms": "Worm activity in moist soil and root zones; dark nutrient-rich vermicastings; zero foliage damage.",
        "damage_risk": "CRITICAL BENEFIT - Aerates soil, improves water infiltration, and recycles soil organic matter.",
        "cultural_actions": [
            "Add organic compost, crop residue, and well-rotted manure to feed soil earthworms.",
            "Practice minimum or conservation tillage to protect worm burrows."
        ],
        "biological_controls": [
            "No control needed. Earthworms are the foundation of living soil fertility."
        ],
        "chemical_interventions": [],
        "preventative_practices": [
            "Avoid synthetic chemical soil drenches and high chemical copper applications that poison earthworms."
        ],
        "extension_escalation": "No escalation needed. Follow organic soil health and vermiculture practices."
    },
    "healthy": {
        "common_name": "Healthy Foliage",
        "scientific_name": "Healthy Plant Tissue",
        "primary_crop": "All Crop Species",
        "swahili_name": "Mmea Wenye Afya",
        "symptoms": "Clean green foliage, vigorous growth, absence of necrotic lesions, pest feeding scars, or chlorosis.",
        "damage_risk": "None - Plant exhibits high photosynthetic vigor.",
        "cultural_actions": [
            "Maintain balanced crop nutrition with soil-test-guided NPK and well-cured compost or farmyard manure.",
            "Ensure regular, adequate soil moisture management and timely weeding during critical growth stages.",
            "Keep records of planting dates, fertilizer applications, and scouting observations."
        ],
        "biological_controls": [
            "Promote soil microbial health by applying mycorrhizal fungi and avoiding broad-spectrum soil sterilants.",
            "Encourage beneficial pollinator insects and predatory arthropods around field borders."
        ],
        "chemical_interventions": [],
        "preventative_practices": [
            "Continue regular weekly scouting in a 'W' pattern across fields to detect early pest or disease arrivals.",
            "Maintain clean border strips and avoid movement of machinery or tools from diseased neighboring plots."
        ],
        "extension_escalation": "No urgent intervention required. Follow standard Good Agricultural Practices (GAP) for your zone."
    },
    "unknown": {
        "common_name": "Unconfirmed Disorder / Low Confidence Diagnosis",
        "scientific_name": "Diagnosis Indeterminate",
        "primary_crop": "Agricultural Crops",
        "swahili_name": "Uchunguzi Haujakamilika",
        "symptoms": "Visual patterns do not meet confidence thresholds for definitive automated diagnosis.",
        "damage_risk": "Unknown - Uncontrolled pests or diseases can escalate rapidly without correct identification.",
        "cultural_actions": [
            "Do NOT apply broad-spectrum chemical pesticides blindly without a verified identification.",
            "Carefully examine upper and lower leaf surfaces with a magnifying glass for tiny mites, thrips, or fungal fruiting bodies.",
            "Take 3-5 sharp, close-up photos in natural diffuse daylight (avoid direct blinding sun or heavy shade).",
            "Collect a fresh leaf sample with both diseased and healthy green tissue, seal in a dry paper bag (not airtight plastic)."
        ],
        "biological_controls": [
            "Safe general preventative: Mild neem oil extract (1-2%) or horticultural soap spray while awaiting diagnosis."
        ],
        "chemical_interventions": [],
        "preventative_practices": [
            "Isolate suspicious plants if only a few are affected to prevent potential viral or bacterial contagion."
        ],
        "extension_escalation": (
            "SAFETY NOTICE: Because the AI confidence is below the safety threshold, chemical intervention is withheld. "
            "Please take the leaf sample or clear photos to your nearest Ward Agricultural Officer, KALRO Centre, "
            "or contact the KALRO Plant Clinic Helpline at 0800 721 741 / WhatsApp +254 711 000 000."
        )
    }
}

def normalize_diagnosis_key(prediction: str, scientific_name: Optional[str] = None) -> str:
    """Matches prediction string to advisory database key."""
    text = f"{prediction} {scientific_name or ''}".lower()

    if "armyworm" in text or "frugiperda" in text or "spodoptera" in text:
        return "fall armyworm"
    elif "stalk borer" in text or "stem borer" in text or "busseola" in text or "chilo" in text:
        return "stalk borer"
    elif "angular" in text or "griseola" in text:
        return "angular leaf spot"
    elif "late blight" in text or "infestans" in text or "phytophthora" in text:
        return "late blight"
    elif "early blight" in text or "alternaria" in text or "target" in text:
        return "early blight"
    elif "slug" in text or "snail" in text or "gastropod" in text:
        return "slugs"
    elif "caterpillar" in text or "viwavi" in text or "larva" in text:
        return "caterpillars"
    elif "beetle" in text or "coleoptera" in text or "mbawakawa" in text:
        return "beetles"
    elif "grasshopper" in text or "locust" in text or "nzige" in text or "panzi" in text:
        return "grasshoppers"
    elif "weevil" in text or "curculio" in text or "dumuzi" in text or "mundu" in text:
        return "weevils"
    elif "earwig" in text or "dermaptera" in text or "kikope" in text:
        return "earwigs"
    elif "ant" in text or "formicidae" in text or "sisimizi" in text:
        return "ants"
    elif "moth" in text or "nondo" in text:
        return "moths"
    elif "bee" in text or "apis" in text or "nyuki" in text:
        return "bees"
    elif "wasp" in text or "nyigu" in text:
        return "wasps"
    elif "earthworm" in text or "lumbric" in text or "mnyoo" in text:
        return "earthworms"
    elif "healthy" in text or "clean" in text:
        return "healthy"
    else:
        return "unknown"

def generate_agronomic_advisory(
    prediction: str,
    scientific_name: Optional[str] = None,
    severity: Optional[str] = None,
    is_unknown: bool = False,
    confidence: float = 1.0,
    threshold: float = 0.40
) -> Dict[str, Any]:
    """
    Generates structured agronomic advisory package conforming to OAN Kenya specification.
    """
    if is_unknown or confidence < threshold or "unknown" in prediction.lower() or prediction == "Error":
        key = "unknown"
    else:
        key = normalize_diagnosis_key(prediction, scientific_name)

    rec = KENYA_AGRI_ADVISORY_DB.get(key, KENYA_AGRI_ADVISORY_DB["unknown"])

    # Adjust severity label
    sev_display = severity or "STAGE_2_MODERATE"
    if key == "healthy":
        sev_display = "HEALTHY"
    elif key == "unknown":
        sev_display = "UNCONFIRMED"

    return {
        "common_name": rec["common_name"],
        "scientific_name": rec["scientific_name"],
        "swahili_name": rec.get("swahili_name", ""),
        "primary_crop": rec["primary_crop"],
        "severity": sev_display,
        "symptoms": rec["symptoms"],
        "damage_risk": rec["damage_risk"],
        "cultural_actions": rec["cultural_actions"],
        "biological_controls": rec["biological_controls"],
        "chemical_interventions": rec["chemical_interventions"],
        "preventative_practices": rec["preventative_practices"],
        "extension_escalation": rec["extension_escalation"],
        "confidence": confidence,
        "is_safe_to_treat": key != "unknown" and len(rec["chemical_interventions"]) > 0
    }
