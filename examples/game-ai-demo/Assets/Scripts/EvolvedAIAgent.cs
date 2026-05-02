using UnityEngine;

/// <summary>
/// Represents an evolved AI agent controlled by an OMNIcode circuit
/// </summary>
public class EvolvedAIAgent : MonoBehaviour
{
    [SerializeField] private int agentId = 0;
    [SerializeField] private float moveSpeed = 5f;
    [SerializeField] private float senseRange = 10f;
    
    // Inputs to the circuit
    private bool canSeeTarget = false;
    private bool obstacleAhead = false;
    private bool healthLow = false;
    
    // Circuit decision output
    private bool shouldAttack = false;
    
    // State
    private float health = 100f;
    private Transform target = null;
    private Vector3 moveDirection = Vector3.zero;
    
    // Circuit reference (would be loaded from C# wrapper)
    private OmnimcodeCircuit circuit = null;
    
    private void Start()
    {
        // Initialize circuit (would load evolved XOR/decision circuit)
        if (circuit == null)
        {
            // Create demo circuit (simple XOR for this example)
            circuit = gameObject.AddComponent<OmnimcodeCircuit>();
        }
    }
    
    private void Update()
    {
        // Sense environment
        SenseEnvironment();
        
        // Evaluate circuit with sensory inputs
        var inputs = new bool[] { canSeeTarget, obstacleAhead, healthLow };
        shouldAttack = circuit.Evaluate(inputs);
        
        // Act based on circuit decision
        Act();
        
        // Display debug info
        DisplayDebugInfo();
    }
    
    private void SenseEnvironment()
    {
        // Check if target in range and visible
        canSeeTarget = false;
        if (target != null)
        {
            float distance = Vector3.Distance(transform.position, target.position);
            if (distance < senseRange)
            {
                RaycastHit hit;
                Vector3 direction = (target.position - transform.position).normalized;
                if (Physics.Raycast(transform.position, direction, out hit, senseRange))
                {
                    if (hit.transform == target)
                    {
                        canSeeTarget = true;
                    }
                }
            }
        }
        
        // Check for obstacles ahead
        obstacleAhead = Physics.Raycast(transform.position, moveDirection, 1f);
        
        // Check health status
        healthLow = health < 30f;
    }
    
    private void Act()
    {
        // Move toward target or away based on circuit decision
        if (shouldAttack && canSeeTarget)
        {
            // Attack mode: move toward target
            Vector3 targetDir = (target.position - transform.position).normalized;
            if (!obstacleAhead)
            {
                transform.position += targetDir * moveSpeed * Time.deltaTime;
            }
            moveDirection = targetDir;
        }
        else
        {
            // Defensive mode: move away
            moveDirection = Random.insideUnitSphere;
            moveDirection.y = 0;
            moveDirection = moveDirection.normalized;
            
            if (!obstacleAhead)
            {
                transform.position += moveDirection * moveSpeed * Time.deltaTime;
            }
        }
    }
    
    private void DisplayDebugInfo()
    {
        Debug.Log($"[Agent {agentId}] Inputs: Target={canSeeTarget} Obstacle={obstacleAhead} Health={healthLow} | Decision: Attack={shouldAttack}");
    }
    
    public void SetTarget(Transform newTarget)
    {
        target = newTarget;
    }
    
    public void TakeDamage(float damage)
    {
        health -= damage;
        if (health <= 0)
        {
            Destroy(gameObject);
        }
    }
    
    public float GetHealth() => health;
    public bool IsAttacking() => shouldAttack;
    public bool CanSeeTarget() => canSeeTarget;
}
