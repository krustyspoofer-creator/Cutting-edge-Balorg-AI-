# Implementation Guidelines for Balorg AI Billing System

## Overview
This document provides technical implementation guidelines for integrating the Balorg AI pricing model into your application.

## Architecture Components

### 1. Subscription Management System

#### Required Database Tables

**users**
```sql
CREATE TABLE users (
    id UUID PRIMARY KEY,
    email VARCHAR(255) UNIQUE NOT NULL,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);
```

**subscriptions**
```sql
CREATE TABLE subscriptions (
    id UUID PRIMARY KEY,
    user_id UUID REFERENCES users(id),
    plan_type VARCHAR(50) NOT NULL, -- 'free', 'starter', 'professional', 'enterprise'
    status VARCHAR(50) NOT NULL, -- 'active', 'cancelled', 'past_due', 'trialing'
    current_period_start TIMESTAMP NOT NULL,
    current_period_end TIMESTAMP NOT NULL,
    trial_end TIMESTAMP,
    cancel_at_period_end BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);
```

**usage_tracking**
```sql
CREATE TABLE usage_tracking (
    id UUID PRIMARY KEY,
    user_id UUID REFERENCES users(id),
    subscription_id UUID REFERENCES subscriptions(id),
    api_calls_count INTEGER DEFAULT 0,
    period_start TIMESTAMP NOT NULL,
    period_end TIMESTAMP NOT NULL,
    created_at TIMESTAMP DEFAULT NOW()
);
```

**discounts**
```sql
CREATE TABLE discounts (
    id UUID PRIMARY KEY,
    user_id UUID REFERENCES users(id),
    discount_type VARCHAR(50) NOT NULL, -- 'student', 'nonprofit', 'early_adopter', 'bulk', 'income_based', 'referral'
    discount_percentage DECIMAL(5,2),
    discount_amount DECIMAL(10,2),
    valid_from TIMESTAMP NOT NULL,
    valid_until TIMESTAMP,
    verification_status VARCHAR(50) DEFAULT 'pending', -- 'pending', 'approved', 'rejected'
    verification_documents JSONB,
    created_at TIMESTAMP DEFAULT NOW()
);
```

### 2. API Rate Limiting

#### Implementation with Redis

```python
import redis
from datetime import datetime, timedelta

class RateLimiter:
    def __init__(self, redis_client, user_id, plan_type):
        self.redis = redis_client
        self.user_id = user_id
        self.limits = {
            'free': 1000,
            'starter': 10000,
            'professional': 100000,
            'enterprise': float('inf')
        }
        self.plan_type = plan_type
        self.limit = self.limits.get(plan_type, 0)
    
    def check_and_increment(self):
        """Check if user is within rate limit and increment counter"""
        key = f"api_calls:{self.user_id}:{datetime.now().strftime('%Y-%m')}"
        current = self.redis.get(key)
        
        if current is None:
            # First call of the month
            self.redis.setex(key, timedelta(days=31), 1)
            return True, 1
        
        current = int(current)
        if current >= self.limit:
            return False, current
        
        self.redis.incr(key)
        return True, current + 1
    
    def get_usage(self):
        """Get current usage for the month"""
        key = f"api_calls:{self.user_id}:{datetime.now().strftime('%Y-%m')}"
        current = self.redis.get(key)
        return int(current) if current else 0
```

### 3. Payment Integration

#### Stripe Integration Example

```python
import stripe
from decimal import Decimal

stripe.api_key = 'your_stripe_secret_key'

class PaymentProcessor:
    
    PRICE_IDS = {
        'starter_monthly': 'price_starter_monthly',
        'starter_annual': 'price_starter_annual',
        'professional_monthly': 'price_professional_monthly',
        'professional_annual': 'price_professional_annual'
    }
    
    def create_subscription(self, user_email, plan_type, billing_cycle, discount_code=None):
        """Create a new subscription for a user"""
        
        # Create or retrieve customer
        customer = stripe.Customer.create(
            email=user_email,
            metadata={'user_id': user_email}
        )
        
        # Get price ID
        price_key = f"{plan_type}_{billing_cycle}"
        price_id = self.PRICE_IDS.get(price_key)
        
        # Apply discount if applicable
        coupon = None
        if discount_code:
            coupon = self.validate_and_get_coupon(discount_code)
        
        # Create subscription
        subscription = stripe.Subscription.create(
            customer=customer.id,
            items=[{'price': price_id}],
            trial_period_days=14,
            coupon=coupon,
            metadata={
                'plan_type': plan_type,
                'billing_cycle': billing_cycle
            }
        )
        
        return subscription
    
    def apply_discount(self, subscription_id, discount_percentage):
        """Apply a discount to an existing subscription"""
        
        # Create a coupon
        coupon = stripe.Coupon.create(
            percent_off=discount_percentage,
            duration='repeating',
            duration_in_months=12
        )
        
        # Update subscription
        subscription = stripe.Subscription.modify(
            subscription_id,
            coupon=coupon.id
        )
        
        return subscription
    
    def validate_and_get_coupon(self, discount_code):
        """Validate discount code and return coupon"""
        discount_map = {
            'STUDENT50': 50,
            'NONPROFIT40': 40,
            'EARLY30': 30,
            'BULK15': 15,
            'BULK25': 25,
            'BULK35': 35
        }
        
        if discount_code in discount_map:
            return stripe.Coupon.create(
                percent_off=discount_map[discount_code],
                duration='repeating',
                duration_in_months=12
            )
        return None
```

### 4. Discount Verification System

```python
class DiscountVerifier:
    
    def verify_student(self, email, student_id_document):
        """Verify student status"""
        # Check for .edu email domain
        if email.endswith('.edu'):
            return True, "Approved via .edu email"
        
        # Manual review required for document verification
        return False, "Document review required"
    
    def verify_nonprofit(self, tax_id, documents):
        """Verify non-profit status"""
        # Integration with IRS database or manual verification
        # This would typically involve checking 501(c)(3) status
        return False, "Manual verification required"
    
    def verify_income_based(self, income_documents):
        """Review income-based discount application"""
        # Confidential manual review process
        return False, "Manual verification required"
    
    def apply_bulk_discount(self, seat_count):
        """Calculate bulk discount based on seat count"""
        if seat_count >= 50:
            return 'Contact sales for custom pricing'
        elif seat_count >= 25:
            return 35
        elif seat_count >= 10:
            return 25
        elif seat_count >= 5:
            return 15
        return 0
```

### 5. Usage Alerts and Notifications

```python
class UsageMonitor:
    
    def __init__(self, user_id, plan_limit):
        self.user_id = user_id
        self.plan_limit = plan_limit
        self.alert_thresholds = [0.8, 0.9, 1.0]
    
    def check_and_alert(self, current_usage):
        """Check usage and send alerts at thresholds"""
        usage_percentage = current_usage / self.plan_limit
        
        for threshold in self.alert_thresholds:
            if usage_percentage >= threshold:
                self.send_alert(usage_percentage, threshold)
    
    def send_alert(self, usage_percentage, threshold):
        """Send usage alert to user"""
        if threshold == 0.8:
            message = f"You've used 80% of your monthly API calls"
        elif threshold == 0.9:
            message = f"You've used 90% of your monthly API calls. Consider upgrading."
        else:
            message = f"You've reached your API call limit. Upgrade to continue."
        
        # Send email notification
        self.send_email_notification(message)
    
    def send_email_notification(self, message):
        """Send email notification (implement with your email service)"""
        pass
```

### 6. Referral Program Implementation

```python
class ReferralProgram:
    
    def __init__(self):
        self.referral_credit = 10.00  # USD
    
    def generate_referral_code(self, user_id):
        """Generate unique referral code for user"""
        import hashlib
        hash_input = f"{user_id}{datetime.now().timestamp()}"
        code = hashlib.md5(hash_input.encode()).hexdigest()[:8].upper()
        return f"REF-{code}"
    
    def apply_referral(self, referrer_id, referee_id):
        """Apply referral credits to both users"""
        # Credit the referrer
        self.add_credit(referrer_id, self.referral_credit)
        
        # Credit the referee
        self.add_credit(referee_id, self.referral_credit)
        
        # Track referral
        self.track_referral(referrer_id, referee_id)
    
    def add_credit(self, user_id, amount):
        """Add credit to user account"""
        # Update user credits in database
        pass
    
    def track_referral(self, referrer_id, referee_id):
        """Track successful referral"""
        # Store referral relationship in database
        pass
```

### 7. Plan Upgrade/Downgrade Logic

```python
class SubscriptionManager:
    
    def upgrade_plan(self, subscription_id, new_plan_type):
        """Upgrade user to higher tier immediately"""
        subscription = stripe.Subscription.modify(
            subscription_id,
            items=[{
                'id': subscription['items']['data'][0].id,
                'price': self.get_price_id(new_plan_type),
            }],
            proration_behavior='always_invoice',
        )
        return subscription
    
    def downgrade_plan(self, subscription_id, new_plan_type):
        """Downgrade user at end of billing period"""
        subscription = stripe.Subscription.modify(
            subscription_id,
            items=[{
                'id': subscription['items']['data'][0].id,
                'price': self.get_price_id(new_plan_type),
            }],
            proration_behavior='none',
            billing_cycle_anchor='unchanged'
        )
        return subscription
    
    def cancel_subscription(self, subscription_id, immediate=False):
        """Cancel subscription"""
        if immediate:
            subscription = stripe.Subscription.delete(subscription_id)
        else:
            subscription = stripe.Subscription.modify(
                subscription_id,
                cancel_at_period_end=True
            )
        return subscription
```

### 8. API Endpoints

#### Required REST API Endpoints

```
GET    /api/v1/pricing/plans              - List all available plans
GET    /api/v1/pricing/plans/:id           - Get specific plan details
POST   /api/v1/subscriptions               - Create new subscription
GET    /api/v1/subscriptions/:id           - Get subscription details
PUT    /api/v1/subscriptions/:id           - Update subscription
DELETE /api/v1/subscriptions/:id           - Cancel subscription
GET    /api/v1/usage/current                - Get current usage stats
POST   /api/v1/discounts/apply             - Apply discount code
POST   /api/v1/discounts/verify            - Submit discount verification
GET    /api/v1/billing/invoices            - List invoices
GET    /api/v1/billing/invoices/:id        - Get specific invoice
POST   /api/v1/referrals/generate          - Generate referral code
POST   /api/v1/referrals/apply             - Apply referral code
```

## Security Considerations

### 1. Data Protection
- Encrypt all payment information in transit and at rest
- PCI DSS compliance for credit card handling
- Never store full credit card numbers
- Use tokenization for payment methods

### 2. Access Control
- Implement proper authentication and authorization
- Rate limiting on API endpoints
- Audit logging for all subscription changes
- Secure webhook endpoints with signature verification

### 3. Fraud Prevention
- Monitor for unusual usage patterns
- Implement CAPTCHA for sign-ups
- Email verification for new accounts
- Transaction monitoring and alerts

## Testing Strategy

### 1. Unit Tests
- Test rate limiting logic
- Test discount calculation
- Test subscription state transitions
- Test payment processing (with mocks)

### 2. Integration Tests
- Test full subscription flow
- Test payment gateway integration
- Test webhook handling
- Test upgrade/downgrade scenarios

### 3. Load Tests
- Test API rate limiting under load
- Test concurrent subscription operations
- Test usage tracking accuracy

## Monitoring and Analytics

### Key Metrics to Track
1. Monthly Recurring Revenue (MRR)
2. Customer Acquisition Cost (CAC)
3. Customer Lifetime Value (LTV)
4. Churn Rate
5. Conversion Rate (Free to Paid)
6. Average Revenue Per User (ARPU)
7. Plan Distribution
8. Discount Usage Rates
9. API Usage Patterns
10. Payment Success/Failure Rates

### Recommended Tools
- Stripe for payment processing
- Redis for rate limiting
- PostgreSQL for data storage
- Prometheus + Grafana for monitoring
- Segment for analytics
- SendGrid/Mailgun for email notifications

## Deployment Checklist

- [ ] Set up payment gateway account
- [ ] Configure database tables
- [ ] Implement rate limiting service
- [ ] Set up email notification service
- [ ] Create pricing page
- [ ] Implement subscription API endpoints
- [ ] Set up webhook handlers
- [ ] Configure monitoring and alerting
- [ ] Test all subscription flows
- [ ] Set up analytics tracking
- [ ] Create admin dashboard
- [ ] Document API for integrations
- [ ] Set up customer support tools
- [ ] Configure automated invoicing
- [ ] Implement usage reporting
- [ ] Test discount verification workflows
- [ ] Set up referral tracking
- [ ] Create terms of service
- [ ] Create privacy policy
- [ ] Launch beta with limited users
- [ ] Monitor and iterate based on feedback

## Support Resources

### Customer Support Scripts
Prepare support team with:
- Plan comparison guides
- Discount eligibility criteria
- Upgrade/downgrade procedures
- Refund policies
- Troubleshooting guides

### Documentation
Create comprehensive docs for:
- Getting started guides
- API documentation
- Billing FAQs
- Integration guides
- Best practices

---

*For technical questions about implementation, contact: dev@balorg-ai.com*
*Last Updated: October 2025*
