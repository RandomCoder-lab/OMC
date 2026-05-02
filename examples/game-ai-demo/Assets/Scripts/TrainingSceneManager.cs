using UnityEngine;
using UnityEngine.UI;
using System.Collections.Generic;

/// <summary>
/// Manages the training scene - evolves circuits and measures performance
/// </summary>
public class TrainingSceneManager : MonoBehaviour
{
    [SerializeField] private int populationSize = 32;
    [SerializeField] private int generationsPerUpdate = 10;
    [SerializeField] private Text fitnessText;
    [SerializeField] private Text generationText;
    [SerializeField] private Button evolveButton;
    [SerializeField] private Slider progressSlider;
    
    private int currentGeneration = 0;
    private float bestFitness = 0f;
    private List<EvolvedAIAgent> population = new List<EvolvedAIAgent>();
    private bool isEvolving = false;
    
    private void Start()
    {
        if (evolveButton != null)
        {
            evolveButton.onClick.AddListener(OnEvolveButtonClicked);
        }
        InitializePopulation();
    }
    
    private void InitializePopulation()
    {
        // Spawn initial population of AI agents
        for (int i = 0; i < populationSize; i++)
        {
            GameObject agentObj = new GameObject($"Agent_{i}");
            agentObj.transform.parent = transform;
            agentObj.transform.position = Random.insideUnitSphere * 10f;
            
            EvolvedAIAgent agent = agentObj.AddComponent<EvolvedAIAgent>();
            population.Add(agent);
        }
    }
    
    private void OnEvolveButtonClicked()
    {
        isEvolving = !isEvolving;
        evolveButton.GetComponentInChildren<Text>().text = isEvolving ? "Stop Evolution" : "Run Evolution";
    }
    
    private void Update()
    {
        if (!isEvolving)
            return;
        
        // Run evolution steps
        for (int i = 0; i < generationsPerUpdate; i++)
        {
            currentGeneration++;
            
            // Evaluate fitness of all agents
            float totalFitness = 0f;
            foreach (var agent in population)
            {
                float fitness = EvaluateAgentFitness(agent);
                totalFitness += fitness;
            }
            
            bestFitness = totalFitness / population.Count;
            
            // Update UI
            if (fitnessText != null)
                fitnessText.text = $"Best Fitness: {bestFitness:F2}%";
            if (generationText != null)
                generationText.text = $"Generation: {currentGeneration}";
            if (progressSlider != null)
                progressSlider.value = bestFitness;
        }
    }
    
    private float EvaluateAgentFitness(EvolvedAIAgent agent)
    {
        // Simple fitness: Did the agent make intelligent decisions?
        // (In real implementation, would measure actual game performance)
        bool canSee = agent.CanSeeTarget();
        bool attacking = agent.IsAttacking();
        float health = agent.GetHealth();
        
        // Reward: Attack when can see, defend when health low
        float fitness = 0f;
        if (canSee && attacking) fitness += 0.5f;
        if (health > 50f) fitness += 0.5f;
        
        return fitness * 100f;
    }
}
