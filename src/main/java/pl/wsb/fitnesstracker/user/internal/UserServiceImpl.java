package pl.wsb.fitnesstracker.user.internal;

import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;
import pl.wsb.fitnesstracker.user.api.User;
import pl.wsb.fitnesstracker.user.api.UserProvider;
import pl.wsb.fitnesstracker.user.api.UserService;

import java.util.List;
import java.util.Optional;

/**
 * Service implementation for user management operations.
 * Provides business logic for creating, retrieving, updating and deleting users.
 * Acts as the business logic layer between the controller and repository.
 */
@Service
@RequiredArgsConstructor
@Slf4j
class UserServiceImpl implements UserService, UserProvider {

    private final UserRepository userRepository;

    /**
     * Creates a new user in the system.
     * User cannot have a database ID at creation time.
     * 
     * @param user the user to create without an ID
     * @return the created user with generated ID
     * @throws IllegalArgumentException if user already has a database ID
     */
    @Override
    public User createUser(final User user) {
        log.info("Creating User {}", user);
        if (user.getId() != null) {
            throw new IllegalArgumentException("User has already DB ID, update is not permitted!");
        }
        return userRepository.save(user);
    }

    /**
     * Retrieves a user by their unique identifier.
     * 
     * @param userId the unique database identifier
     * @return Optional containing the user if found, or empty Optional if not
     */
    @Override
    public Optional<User> getUser(final Long userId) {
        return userRepository.findById(userId);
    }

    /**
     * Retrieves a user by their email address.
     * Email addresses are unique in the system.
     * 
     * @param email the email address to search for
     * @return Optional containing the user if found, or empty Optional if not
     */
    @Override
    public Optional<User> getUserByEmail(final String email) {
        return userRepository.findByEmail(email);
    }

    /**
     * Retrieves all users from the system.
     * 
     * @return list of all users, or empty list if no users exist
     */
    @Override
    public List<User> findAllUsers() {
        return userRepository.findAll();
    }

    /**
     * Updates an existing user's information.
     * User must have a database ID (must already exist in the system).
     * 
     * @param user the user with updated information including ID
     * @return the updated user
     * @throws IllegalArgumentException if user doesn't have a database ID
     */
    public User updateUser(final User user) {
        log.info("Updating User {}", user);
        if (user.getId() == null) {
            throw new IllegalArgumentException("User must have a DB ID to update!");
        }
        return userRepository.save(user);
    }

    /**
     * Deletes a user from the system.
     * 
     * @param userId the unique identifier of the user to delete
     */
    public void deleteUser(final Long userId) {
        log.info("Deleting User with ID {}", userId);
        userRepository.deleteById(userId);
    }

}